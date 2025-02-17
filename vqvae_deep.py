import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

import distributed as dist_fn


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, in_channel, channel, style_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, channel, 3, padding=1)
        self.conv2 = nn.Conv2d(channel, in_channel, 1)
        self.norm1 = AdaIN(style_dim, in_channel)
        self.norm2 = AdaIN(style_dim, channel)

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input, s):
        out = self.norm1(input, s)
        out = self.conv1(F.relu_(out))
        out = self.norm2(out, s)
        out = self.conv2(F.relu_(out))
        out += input

        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        blocks = []

        def down4(in_channel):
            return [nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel, channel, 3, padding=1)
                    ]

        def down2(in_channel):
            return [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]
        if stride == 8:
            blocks.extend(down4(in_channel))
            blocks.extend(down4(channel))
        if stride == 6:
            blocks.extend(down2(in_channel))
            blocks.extend(down4(channel))
        elif stride == 4:
            blocks.extend(down4(in_channel))
        elif stride == 2:
            blocks.extend(down2(in_channel))

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, channel, style_dim , n_res_block, n_res_channel, stride
    ):
        super().__init__()
        self.style = True if style_dim > 1 else False
        blocks = []
        self.conv1 = nn.Conv2d(in_channel, channel, 3, padding=1)
        for i in range(n_res_block):
            # blocks.append(AdaptiveInstanceNorm2d(channel))
            if style_dim <= 0:
                blocks.append(ResBlock(channel, n_res_channel))
            else:
                blocks.append(AdainResBlk(channel, n_res_channel, style_dim))

        self.relu = nn.ReLU(inplace=True)
        # blocks.append(AdaptiveInstanceNorm2d(channel))

        def up4(channel):
            return [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]

        def up2(channel):
            return [nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)]

        up_sample=[]
        if stride == 8:
            up_sample.extend(up4(channel))
            up_sample.extend(up4(out_channel))
        elif stride == 6:
                up_sample.extend(up4(channel))
                up_sample.extend(up2(out_channel))
        elif stride == 4:
            up_sample.extend(up4(channel))
        elif stride == 2:
            up_sample.extend(up2(channel))
        self.up_sample = nn.Sequential(*up_sample)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input, s = None):
        out = self.conv1(input)
        if not self.style:
            out = self.blocks(out)
        else:
            for adainBlk in self.blocks:
                out = adainBlk(out, s)
        out = self.up_sample(self.relu(out))
        return out




class VQVAE_Deep(nn.Module):
    def __init__(
            self,
            in_channel=3,
            channel=256,
            n_res_block=6,
            n_res_channel=128,
            embed_dim=256,
            n_embed=512,
            decay=0.99,
            out_channel=3,
            style_dim = 2048
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=6)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, -1, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        )

        self.dec = Decoder(
            embed_dim + embed_dim,
            out_channel,
            channel,
            style_dim,
            n_res_block,
            n_res_channel,
            stride=6,
        )

        self.embed_dim = 2 * embed_dim

    def forward(self, input):
        enc_b, enc_t = self.encode(input)
        quant_t, quant_b, diff, _, _ = self.quantize(enc_b, enc_t)
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.decode(quant)
        return dec, diff, quant

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        return enc_b, enc_t

    def quantize(self, enc_b, enc_t):
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    # def decode(self, quant_t, quant_b):
    #
    #     dec = self.dec(quant)
    #     dec2 = self.dec_ir(quant.detach()).expand(-1, 3, -1, -1)
    #     return dec, dec2

    def decode(self, quant, style):
        return self.dec(quant, style)

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
