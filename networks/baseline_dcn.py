import torch.nn as nn
import torch
affine_par = True
import functools

from inplace_abn import InPlaceABN, InPlaceABNSync
from utils.pyt_utils import load_model
from ops.dcn import DeformConv
from mmcv.cnn import constant_init

BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, DCN=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        if not DCN:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv2 = DeformConv(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
            self.conv2_offset = nn.Conv2d(planes, 18, kernel_size=3, stride=stride, padding=dilation, dilation=dilation)
            constant_init(self.conv2_offset, val=0)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.DCN = DCN

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.DCN:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
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
    def __init__(self, in_channels, out_channels, num_classes):
        super(HeadModule, self).__init__()

        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), BatchNorm2d(inter_channels))
        self.convb = nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False)
        self.sync_bn_convb = BatchNorm2d(inter_channels)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        output = self.conva(x)
        output = self.convb(output)
        output = self.sync_bn_convb(output)
        output = torch.cat([x, output], 1)
        output = self.bottleneck(output)

        return output

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, frozen_stages, bn_frozen, criterion):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.frozen_stages = frozen_stages
        self.bn_frozen=bn_frozen
        self.criterion = criterion

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

        self.layer1 = self._make_layer(block, 64, layers[0], DCN=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, DCN=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, DCN=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, DCN=True)

        self.head = HeadModule(2048, 512, num_classes)
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, DCN=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, DCN=DCN))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, DCN=DCN))

        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
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
        outs = [x, x_dsn]

        if self.criterion is not None and labels is not None:
            return self.criterion(outs, labels)
        else:
            return outs

    def train(self, mode=True):
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
            print('[CHECK] Frozen Nothing')

def Seg_Model(num_classes=21, num_layers=101, frozen_stages=-1, bn_frozen=False, pretrained_model=None, criterion=None,
              **kwargs):
    layers = []
    if num_layers == 50:
        layers = [3, 4, 6, 3]
    elif num_layers == 101:
        layers = [3, 4, 23, 3]

    model = ResNet(Bottleneck, layers, num_classes, frozen_stages=frozen_stages, bn_frozen=bn_frozen, criterion=criterion)
    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model
