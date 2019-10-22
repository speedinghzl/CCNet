import torch.nn as nn
import torch
import numpy as np
affine_par = True
import functools

from encoding.nn import syncbn
from libs import InPlaceABN, InPlaceABNSync

from ops.dcn import GeneralizedAttention

# BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
# BatchNorm2d = functools.partial(syncbn.BatchNorm2d)
BatchNorm2d = functools.partial(nn.BatchNorm2d)
def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 use_non_local=False, **non_local_params):
        super(Bottleneck, self).__init__()


        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)

        self.use_non_local = use_non_local
        if self.use_non_local:
            non_local_position = non_local_params['non_local_position']
            non_local_params.pop('non_local_position', None)
            # get non local params
            non_local_planes = (planes * self.expansion) if (non_local_position == 'after_relu') else planes

            self.non_local_block = GeneralizedAttention(in_dim=non_local_planes,
                                                        out_dim=non_local_planes,
                                                        share_with_conv2=(non_local_position == 'conv2'),
                                                        conv2_weight=self.conv2.weight,
                                                        conv2_bias=self.conv2.bias,
                                                        conv2_stride=stride,
                                                        conv2_dilation=dilation,
                                                        **non_local_params)

        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.use_non_local:
            out = self.non_local_block(out)
        else:
            out = self.conv2(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out

class HeadModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, use_non_local, **non_local_params):
        super(HeadModule, self).__init__()

        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   syncbn.BatchNorm2d(inter_channels))
        self.convb = nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False)
        self.sync_bn_convb = syncbn.BatchNorm2d(inter_channels)

        # self.conv1 = nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False)

        self.use_non_local = use_non_local
        if self.use_non_local:
            non_local_params.pop('non_local_position', None)
            # non_local_params['kv_stride'] = 2
            self.non_local_block = GeneralizedAttention(in_dim=inter_channels,
                                                        out_dim=inter_channels,
                                                        share_with_conv2=True,
                                                        conv2_weight=self.convb.weight,
                                                        conv2_bias=self.convb.bias,
                                                        conv2_stride=1,
                                                        **non_local_params)


        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            syncbn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x):

        output = self.conva(x)
        if self.use_non_local:
            output = self.non_local_block(output)
        else:
            output = self.convb(output)
        output = self.sync_bn_convb(output)
        output = torch.cat([x, output], 1)
        output = self.bottleneck(output)


        return output

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes,
                 use_non_local, num_non_local_block, use_non_local_in_head, frozen_stages, bn_frozen, **non_local_params):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.frozen_stages = frozen_stages
        self.bn_frozen=bn_frozen

        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change

        self.layer1 = self._make_layer(block, 64, layers[0], use_non_local=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_non_local=False)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2,
                                       use_non_local=use_non_local,
                                       num_non_local_block=num_non_local_block-layers[3],
                                       **non_local_params)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4,
                                       use_non_local=use_non_local,
                                       num_non_local_block=num_non_local_block,
                                       **non_local_params)

        self.head = HeadModule(2048, 512, num_classes, use_non_local=use_non_local_in_head, **non_local_params)

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            syncbn.BatchNorm2d(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )


    def _make_layer(self, block, planes, blocks,
                    stride=1,
                    dilation=1,
                    use_non_local=False,
                    num_non_local_block=999,
                    **non_local_params):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []

        use_non_local_t = use_non_local if blocks <= num_non_local_block else False
        layers.append(block(self.inplanes,
                            planes,
                            stride,dilation=dilation,
                            downsample=downsample,
                            use_non_local=use_non_local_t,
                            **non_local_params))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            use_non_local_t = use_non_local if blocks - i <= num_non_local_block else False
            layers.append(block(self.inplanes,
                                planes,
                                dilation=dilation,
                                use_non_local=use_non_local_t,
                                **non_local_params))

        return nn.Sequential(*layers)

    def forward(self, x, *kwargs):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)
        x = self.head(x)
        return [x, x_dsn]

    def train(self, mode=True):
        # import IPython
        # IPython.embed()
        super(ResNet, self).train(mode)
        if self.bn_frozen:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for params in m.parameters():
                        params.requires_grad = False

        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False
        else:
            print('frozen nothing')

def Res_Deeplab(num_classes=21, num_layers=101,
                use_non_local=False, num_non_local_block=999,
                use_non_local_in_head=False, frozen_stages=-1, bn_frozen=False, **non_local_params):
    layers = []

    if num_layers == 50:
        layers = [3, 4, 6, 3]
    elif num_layers == 101:
        layers = [3, 4, 23, 3]

    model = ResNet(Bottleneck, layers, num_classes,
                   use_non_local=use_non_local,
                   num_non_local_block=num_non_local_block,
                   use_non_local_in_head=use_non_local_in_head,
                   frozen_stages=frozen_stages,
                   bn_frozen=bn_frozen,
                   **non_local_params)

    return model
