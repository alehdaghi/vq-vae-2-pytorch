import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from torch.nn import init
from torch.nn import functional as F
import copy
from torch.nn.utils import spectral_norm


from vqvae import VQVAE, Encoder
from vqvae_deep import VQVAE_Deep


def compute_mask(feat):
    batch_size, fdim, h, w = feat.shape
    norms = torch.norm(feat, p=2, dim=1).view(batch_size, h * w)

    norms -= norms.min(dim=-1, keepdim=True)[0]
    norms /= norms.max(dim=-1, keepdim=True)[0] + 1e-12
    mask = norms.view(batch_size, 1, h, w)

    return mask.detach()

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

def ConvModule(c_in, c_out, k = 1, is_init_kaiming = False ):
    conv = nn.Conv2d(c_in, c_out, k)
    if is_init_kaiming:
        conv.apply(weights_init_kaiming)
    return nn.Sequential(conv,
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True)
    )

def LinearModule(d_in, d_out):
    fc = nn.Linear(d_in, d_out)
    init.normal_(fc.weight, std=0.001)
    init.constant_(fc.bias, 0)
    return fc


class embed_net(nn.Module):
    def __init__(self, class_num, no_local='on', gm_pool='on', arch='resnet50', camera_num=6, part = False):
        self.part = part
        super(embed_net, self).__init__()

        if arch == 'resnet50':
            resnet = torchvision.models.resnet50(pretrained=True)
            resnet.layer4[0].conv2.stride = (1, 1)
        elif arch == 'resnet34':
            resnet = torchvision.models.resnet34(pretrained=True)
            resnet.layer4[0].conv1.stride = (1, 1)
        else:
            resnet = torchvision.models.resnet18(pretrained=True)
            resnet.layer4[0].conv1.stride = (1, 1)
        # avg pooling to global pooling

        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.base_resnet = nn.Sequential(resnet.layer2,
                                          nn.Dropout2d(0.05),
                                          resnet.layer3,
                                          nn.Dropout2d(0.05),
                                          resnet.layer4,
                                          nn.Dropout2d(0.1)
                                          )

        if arch == 'resnet50':
            self.pool_dim = 2048
        else:
            self.pool_dim = 512

        if part:
            self.local_conv_list = nn.ModuleList()
            for _ in range(6):
                self.local_conv_list.append(ConvModule(self.pool_dim, 512, is_init_kaiming=True))

            self.fc_list = nn.ModuleList()
            for _ in range(6):
                self.fc_list.append(LinearModule(512, class_num))
            self.pool_dim = 6 * 512



        self.thermal_module = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool,  # no relu
            resnet.layer1)


        self.visible_module = copy.deepcopy(self.thermal_module)
        self.z_module = copy.deepcopy(self.thermal_module)
        self.l2norm = Normalize(2)

        self.bottleneck = nn.BatchNorm1d(self.pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        nn.init.constant_(self.bottleneck.bias, 0)

        # self.drop = nn.Dropout(p=0.2, inplace=True)

        self.classifier = nn.Linear(self.pool_dim, class_num, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool

    def parameters_hook(self, grad):
        print("grad:", grad)

    def forward(self, xRGB, xIR, xZ=None, modal=0, with_feature=False, with_camID=False):
        if modal == 0:
            x1 = self.visible_module(xRGB) if xRGB is not None else self.z_module(xZ)
            x2 = self.thermal_module(xIR)
            x = torch.cat((x1, x2), 0)
            view_size = 2
        elif modal == 1:
            x = self.visible_module(xRGB)
        elif modal == 2:
            x = self.thermal_module(xIR)
        elif modal == 3:
            x = self.z_module(xZ)

        x3 = self.base_resnet[0:3](x)  # layer2 : layer3
        x4 = self.base_resnet[3:5](x3)
        # x = self.base_resnet.resnet_part2[1](x)  # layer3
        # x3 = x
        # x = self.base_resnet.resnet_part2[2](x)  # layer4

        person_mask = compute_mask(x4)

        if self.part:
            local_feat_list = []
            logits_list = []
            p = 10  # regDB: 10.0    SYSU: 3.0
            local_6_feat_tensor = (F.adaptive_avg_pool2d(x4 ** p + 1e-12, (6, 1)) ** (1 / p)).unsqueeze(dim=-1)
            for i in range(6):
                local_feat = self.local_conv_list[i](local_6_feat_tensor[:, :, i])
                # shape [N, c]
                local_feat_list.append(local_feat.squeeze())
                logits_list.append(self.fc_list[i](local_feat.squeeze()))

            feat_all = torch.cat(local_feat_list, dim=1)
            if self.training:
                return local_feat_list, logits_list, feat_all
            else:
                return self.l2norm(feat_all), self.l2norm(feat_all)

        else :
            feat_pool = self.gl_pool(x4, self.gm_pool)
            feat = self.bottleneck(feat_pool)
            # feat = self.drop(feat)

            if with_feature:
                return feat_pool, self.classifier(feat), x4 ,person_mask, x3

            if not self.training:
                return self.l2norm(feat), self.l2norm(feat_pool)
            return feat_pool, self.classifier(feat), #x, person_mask

    @staticmethod
    def gl_pool(x, gm_pool):
        if gm_pool == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)
        else:
            x_pool = F.adaptive_avg_pool2d(x, (1, 1))
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        return x_pool

    def getPoolDim(self):
        return self.pool_dim



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm, activ):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params

def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]



class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm, activation):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer

        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        self.norm = None

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class ModelAdaptive(nn.Module):
    def __init__(self, class_num=395, no_local='on', gm_pool='on', arch='resnet18', camera_num=6):
        super(ModelAdaptive, self).__init__()
        self.person_id = embed_net(class_num, no_local, gm_pool, arch)

        # self.camera_id = Camera_net(camera_num, arch)
        self.fusion = Non_local(128, 1)
        self.adaptor = VQVAE()

        self.style_dim = 128

        self.encoder_s = nn.Sequential(Encoder(3, self.style_dim, 3, 32, stride=2),
                                       Encoder(self.style_dim, self.style_dim, 3, 32, stride=2))

        self.conv1 = spectral_norm(nn.Conv2d(self.style_dim, self.style_dim, kernel_size=1, stride=1, padding=0))
        self.conv2 = spectral_norm(
            nn.Conv2d(self.style_dim, self.style_dim, kernel_size=1, stride=1, padding=0))

        self.resblocks = nn.Sequential(
            ResidualBlock(self.style_dim, self.style_dim),
            ResidualBlock(self.style_dim, self.style_dim),
        )

        # self.upsample_t = nn.Sequential(
        #     nn.ConvTranspose2d(self.person_id.pool_dim, self.person_id.pool_dim, 4, stride=2, padding=1)
        # )

        # self.mlp = MLP(self.person_id.pool_dim, get_num_adain_params(self.adaptor), 256, 1, norm='none', activ='relu')
        # self.discriminator = Discriminator()

    def encode_person(self, rgb):
        feat, score, feat2d, actMap, x3 = self.person_id(xRGB=rgb, xIR=None, modal=1, with_feature=True)
        return feat, score, feat2d, actMap, x3

    def encode_style(self, rgb):
        return self.encoder_s(rgb)


    def encode_content(self, img):
        quant_t, quant_b, diff, _, _ = self.adaptor.encode(img)
        upsample_t = self.adaptor.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        return quant, diff

    def fuse(self, content, style):

        c = self.conv1(content)
        f = self.fusion(c, style)
        f = self.resblocks(f) + f
        newC = self.conv2(f)
        return newC

    def decodeWithStyle(self, content, style):
        self.fusion(content, style)

    def decodeWithoutStyle(self, content):
        return self.adaptor.decode(content)

    def decode(self, content):
        return self.adaptor.decode(content)


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=4):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, c, s):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = c.size(0)
        g_s = self.g(s).view(batch_size, self.inter_channels, -1)
        g_s = g_s.permute(0, 2, 1)

        theta_c = self.theta(c).view(batch_size, self.inter_channels, -1)
        theta_c = theta_c.permute(0, 2, 1)
        phi_s = self.phi(s).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_c, phi_s)
        N = f.size(-1)
        f_div_C = torch.nn.functional.softmax(f / N, dim=-1)
        # f_div_C = f / N

        y = torch.matmul(f_div_C, g_s)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *c.size()[2:])
        W_y = self.W(y)
        z = W_y + c

        return z


class ModelAdaptive_Deep(nn.Module):
    def __init__(self, class_num=395, adaptor=None, arch='resnet18'):
        super(ModelAdaptive_Deep, self).__init__()
        self.person_id = embed_net(class_num, 'off', 'off', arch)

        # self.camera_id = Camera_net(camera_num, arch)
        self.fusion1 = Non_local(256, 1)
        self.fusion2 = Non_local(256, 1)

        self.adaptor = VQVAE_Deep() if adaptor is None else adaptor



        self.style_dim = 256


        self.conv1 = spectral_norm(nn.Conv2d(512, self.style_dim, kernel_size=1, stride=2, padding=0))
        self.conv2 = spectral_norm(
            nn.ConvTranspose2d(self.style_dim, self.style_dim, kernel_size=4, stride=2, padding=1))

        self.resblocks1 = nn.Sequential(
            ResidualBlock(self.style_dim, self.style_dim),
            ResidualBlock(self.style_dim, self.style_dim),
        )
        self.resblocks2 = nn.Sequential(
            ResidualBlock(self.style_dim, self.style_dim),
            ResidualBlock(self.style_dim, self.style_dim),
        )

        # self.upsample_t = nn.Sequential(
        #     nn.ConvTranspose2d(self.person_id.pool_dim, self.person_id.pool_dim, 4, stride=2, padding=1)
        # )

        # self.mlp = MLP(self.person_id.pool_dim, get_num_adain_params(self.adaptor), 256, 1, norm='none', activ='relu')
        # self.discriminator = Discriminator()

    def encode_person(self, rgb):
        feat, score, feat2d, actMap, x3 = self.person_id(xRGB=rgb, xIR=None, modal=1, with_feature=True)
        return feat, score, feat2d, actMap, x3

    def encode_style(self, rgb):
        return self.encoder_s(rgb)


    def encode_content(self, img):
        enc_b, enc_t = self.adaptor.encode(img)
        return enc_b, enc_t

    def quantize_content(self, enc_b, enc_t):
        quant_t, quant_b, diff, _, _ = self.adaptor.quantize(enc_b, enc_t)
        upsample_t = self.adaptor.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        return quant, diff

    def fuse(self, cb, ct, sb, st):
        f = self.fusion1(cb, sb.detach())
        cb = self.resblocks1(f) + f + cb
        f = self.fusion2(ct, self.conv1(st.detach()))
        ct = self.resblocks2(f) + f + ct
        return cb, ct

    def encAndDec(self, img):
        enc_b, enc_t = self.encode_content(img)
        enc_b_f, enc_t_f = enc_b, enc_t  # model.fuse(enc_b, rgb_t, feat2d_x3[bs:] , feat2d[bs:])
        img_content, _ = self.quantize_content(enc_b_f, enc_t_f)
        rec = self.decode(img_content).expand(-1, 3, -1, -1)
        return rec

    def decodeWithStyle(self, content, style):
        self.fusion(content, style)

    def decodeWithoutStyle(self, content):
        return self.adaptor.decode(content)

    def decode(self, content):
        return self.adaptor.decode(content)
