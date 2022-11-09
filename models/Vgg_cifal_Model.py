import math
import torch
import torch.nn as nn


class Vgg_net(nn.Module):
    def __init__(self, depth=11, num_classes=10, init_weights=True, cfg=None):
        super(Vgg_net, self).__init__()

        self.default_cfg = {
            11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
            13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
            16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
            19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
        }

        if cfg is None:
            cfg = self.default_cfg[depth]

        self.feature = self.make_layers(cfg, True)
        self.activations = []

        in_feature = cfg[-1]
        self.classifier = nn.Linear(in_features=in_feature, out_features=num_classes, bias=True)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        # x = nn.AvgPool2d(kernel_size=1, stride=1)(x)
        x = nn.AvgPool2d(2)(x)
        x = torch.flatten(x, start_dim=1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # print(m.kernel_size, m.out_channels)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg11(num_classes=10, cfg=None):
    model = Vgg_net(depth=11, num_classes=num_classes, cfg=cfg)
    return model


def vgg13(num_classes=10, cfg=None):
    model = Vgg_net(depth=13, num_classes=num_classes, cfg=cfg)
    return model


def vgg16(num_classes=10, cfg=None):
    model = Vgg_net(depth=16, num_classes=num_classes, cfg=cfg)
    return model


def vgg19(num_classes=10, cfg=None):
    model = Vgg_net(depth=19, num_classes=num_classes, cfg=cfg)
    return model