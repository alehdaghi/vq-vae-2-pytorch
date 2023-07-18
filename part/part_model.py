import einops
import torch
import torch.nn as nn
from torch.nn import init

import torchvision
import copy
import torch.nn.functional as F

from model import weights_init_kaiming, weights_init_classifier, Normalize, compute_mask
from part.modules.bn import InPlaceABNSync
from part.part_detector import PSPModule, Edge_Module, Decoder_Module


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=4):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

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

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z



class ShallowModule(nn.Module):
    def __init__(self, arch='resnet50'):
        super(ShallowModule, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet_part1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool,  # no relu
            resnet.layer1)

    def forward(self, x):
        x = self.resnet_part1(x)
        return x

class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.resnet_part2 = nn.Sequential(resnet.layer2,
                                          # nn.Dropout2d(0.05),
                                          resnet.layer3,
                                          # nn.Dropout2d(0.05),
                                          resnet.layer4,
                                          # nn.Dropout2d(0.1)
                                          )

    def forward(self, x):
        x = self.resnet_part2(x)
        return x

class embed_net2(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50',
                 separate_batch_norm=False, use_contrast=False):
        super(embed_net2, self).__init__()

        self.thermal_module = ShallowModule(arch=arch)
        self.visible_module = ShallowModule(arch=arch)
        self.gray_module = ShallowModule(arch=arch)


        self.base_resnet = base_resnet(arch=arch)
        self.non_local = no_local
        if self.non_local =='on':
            layers=[3, 4, 6, 3]
            non_layers=[0,2,3,0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        if arch == 'resnet50':
            self.pool_dim = 2048
        else:
            self.pool_dim = 512

        self.use_contrast = use_contrast


        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(self.pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.part_num = 7
        self.classifier = nn.Linear(self.pool_dim + (self.part_num - 1) * 256, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool



        self.maskGen = nn.Sequential(nn.Conv2d(self.part_num, 128, kernel_size=3, padding=1, stride=2, bias=False),
                                     nn.Conv2d(128, self.part_num, kernel_size=3, padding=1, stride=2, bias=False),
                                     nn.Sigmoid(),
                                     nn.Softmax())
        self.part = PartModel(self.part_num)
        self.part_descriptor = nn.Linear(self.pool_dim, 256, bias=False)
        self.pool_dim += (self.part_num - 1) * 256

    def forward(self, xRGB, xIR, xZ=None, modal=0, with_feature = False):
        if modal == 0:
            x1 = self.visible_module(xRGB) if xRGB is not None else self.gray_module(xZ)
            x2 = self.thermal_module(xIR)
            x = torch.cat((x1, x2), 0)
            if xZ is not None and xRGB is not None:
                x3 = self.gray_module(xZ)
                x = torch.cat((x, x3), 0)
        elif modal == 1:
            x = self.visible_module(xRGB)
        elif modal == 2:
            x = self.thermal_module(xIR)
        elif modal == 3:
            x = self.gray_module(xZ)

        x1 = x
        # shared block
        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.resnet_part2[0])):
                x = self.base_resnet.resnet_part2[0][i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            x2 = x
            # Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.resnet_part2[1])):
                x = self.base_resnet.resnet_part2[1][i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            x3 = x #layer
            # Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.resnet_part2[2])):
                x = self.base_resnet.resnet_part2[2][i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
            x4 = x
        else:
            x = self.base_resnet(x)

        b, c, h, w = x.shape
        if self.gm_pool  == 'on':

            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
            x = x.view(b,c, h, w)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat_g = self.bottleneck(x_pool)

        part, partsFeat = self.part(x, x1, x2, x3)

        # return
        part_masks = F.softmax(F.avg_pool2d(part[0][1] + part[0][1], kernel_size=(4,4)))

        maskedFeat = torch.einsum('brhw, bchw -> brc', part_masks[:,1:], x)
        maskedFeat /= einops.reduce(part_masks[:, 1:], 'b r h w -> b r 1', 'sum') + 1e-7

        feats = [feat_g]
        for i in range(1, self.part_num): # 0 is background!
            mask = part_masks[:, i:i + 1, :, :]
            feat = mask * x
            feat = F.avg_pool2d(feat, feat.size()[2:])
            feat = feat.view(feat.size(0), -1)
            feats.append(self.part_descriptor(feat))
        # feats.append(feat_g)
        feats = torch.cat(feats, 1)

        if self.training:
            masks = part_masks.view(b, self.part_num, w * h)
            loss_reg = torch.bmm(masks, masks.permute(0, 2, 1))
            loss_reg = torch.triu(loss_reg, diagonal=1).sum() / (b * self.part_num * (self.part_num - 1) / 2)
            return feats, self.classifier(feats), part, loss_reg, maskedFeat, part_masks
        else:
            return self.l2norm(x_pool), self.l2norm(feats)



class PartModel(nn.Module):
    def __init__(self,  num_part):
        super(PartModel, self).__init__()
        self.context_encoding = PSPModule(2048, 512)

        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_part)

        self.fushion = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_part, kernel_size=1, padding=0, dilation=1, bias=True)
        )

    def forward(self, x, x1, x2, x3):
        x = self.context_encoding(x)
        parsing_result, parsing_fea = self.decoder(x, x1)
        # Edge Branch
        edge_result, edge_fea = self.edge(x1, x2, x3)
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)
        fusion_result = self.fushion(x)
        return [[parsing_result, fusion_result], [edge_result]], x
