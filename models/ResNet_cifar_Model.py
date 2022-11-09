import torch.nn as nn
from utils.Channel_selection import channel_selection

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, cfg=None):
        super(BasicBlock, self).__init__()
        # ---------------------------------------------
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.select = channel_selection(in_channel)

        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)

        # ---------------------------------------------
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)

        # ---------------------------------------------
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, cfg=None):
        super(Bottleneck, self).__init__()
        # ---------------------------------------------
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.select = channel_selection(in_channel)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels=in_channel if cfg is None else cfg[0],
                               out_channels=out_channel if cfg is None else cfg[1],
                               kernel_size=1, stride=1, bias=False)

        # ---------------------------------------------
        self.bn2 = nn.BatchNorm2d(out_channel if cfg is None else cfg[1])
        self.relu2 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channel if cfg is None else cfg[1],
                               out_channels=out_channel if cfg is None else cfg[2],
                               kernel_size=3, stride=stride, bias=False, padding=1)

        # ---------------------------------------------
        self.bn3 = nn.BatchNorm2d(out_channel if cfg is None else cfg[2])
        self.relu3 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=out_channel if cfg is None else cfg[2],
                               out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)

        # ---------------------------------------------
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, cfg=None):
        self.in_channel = 16
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1,
                                       cfg=None if cfg is None else cfg[0: 3 * layers[0]])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       cfg=None if cfg is None else cfg[3 * layers[0]: 6 * layers[1]])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       cfg=None if cfg is None else cfg[6 * layers[1]: 9 * layers[2]])

        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.select = channel_selection(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, blocks, stride=1, cfg=None):
        downsample = None
        if stride != 1 or self.in_channel != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.in_channel, channels, stride, downsample,
                            cfg=None if cfg is None else cfg[0: 3]))

        self.in_channel = channels * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.in_channel, channels,
                                cfg=None if cfg is None else cfg[3*i: 3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


"""
    3n*2+2(BasicBlock)
    3n*3+2(Bottleneck)
"""


def resnet20(num_classes=10, cfg=None):
    n = 3
    model = ResNet(BasicBlock, [n, n, n], num_classes=num_classes, cfg=cfg)
    return model


def resnet32(num_classes=10, cfg=None):
    n = 6
    model = ResNet(BasicBlock, [n, n, n], num_classes=num_classes, cfg=cfg)
    return model


def resnet56(num_classes=10, cfg=None):
    n = 6
    model = ResNet(Bottleneck, [n, n, n], num_classes=num_classes, cfg=cfg)
    return model


def resnet101(num_classes=10, cfg=None):
    n = 11
    model = ResNet(Bottleneck, [n, n, n], num_classes=num_classes, cfg=cfg)
    return model


