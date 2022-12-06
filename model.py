import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from torch.nn import init
from torch.nn import functional as F
import copy

from vqvae import VQVAE


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



class embed_net(nn.Module):
    def __init__(self, class_num, no_local='on', gm_pool='on', arch='resnet50', camera_num=6):
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

        self.thermal_module = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool,  # no relu
            resnet.layer1)


        self.visible_module = copy.deepcopy(self.thermal_module)
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
            x1 = self.visible_module(xRGB)
            x2 = self.thermal_module(xIR)
            x = torch.cat((x1, x2), 0)
            view_size = 2
        elif modal == 1:
            x = self.visible_module(xRGB)
        elif modal == 2:
            x = self.thermal_module(xIR)
        elif modal == 3:
            x = self.z_module(xZ)

        x = self.base_resnet(x)  # layer2
        # x = self.base_resnet.resnet_part2[1](x)  # layer3
        # x3 = x
        # x = self.base_resnet.resnet_part2[2](x)  # layer4

        # person_mask = self.compute_mask(x)
        feat_pool = self.gl_pool(x, self.gm_pool)
        feat = self.bottleneck(feat_pool)
        # feat = self.drop(feat)

        if with_feature:
            return feat_pool, self.classifier(feat), x #, person_mask

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
    def __init__(self, class_num, no_local='on', gm_pool='on', arch='resnet18', camera_num=6):
        super(ModelAdaptive, self).__init__()
        self.person_id = embed_net(class_num, no_local, gm_pool, arch)
        # self.camera_id = Camera_net(camera_num, arch)
        self.adaptor = VQVAE()
        # self.mlp = MLP(self.person_id.pool_dim, get_num_adain_params(self.adaptor), 256, 1, norm='none', activ='relu')
        # self.discriminator = Discriminator()
