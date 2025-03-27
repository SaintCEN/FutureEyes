import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pydensecrf import densecrf

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.activation(x)

class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, skip_connection_type='conv'):
        super().__init__()
        self.skip_connection_type = skip_connection_type

        self.conv1 = SeparableConv2d(in_channels, out_channels, stride=stride, dilation=dilation)
        self.conv2 = SeparableConv2d(out_channels, out_channels, dilation=dilation)
        self.conv3 = SeparableConv2d(out_channels, out_channels, dilation=dilation)

        if skip_connection_type == 'conv':
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        if self.skip_connection_type == 'conv':
            x += residual
        elif self.skip_connection_type == 'sum':
            x = x + residual
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super().__init__()
        modules = []

        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        # Atrous convolutions
        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))

        # Image pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            if isinstance(conv[0], nn.AdaptiveAvgPool2d):
                y = conv(x)
                y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)
                res.append(y)
            else:
                res.append(conv(x))
        x = torch.cat(res, dim=1)
        return self.project(x)


class DeepLabv3Plus(nn.Module):
    def __init__(self, backbone='xception', num_classes=2, output_stride=16):
        super().__init__()
        if backbone == 'xception':
            self.backbone = self._build_xception(output_stride)
            aspp_channels = 2048
            low_level_channels = 256
        else:
            raise NotImplementedError

        self.aspp = ASPP(aspp_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

        self._init_weights()

    def _build_xception(self, output_stride):
        # 完整Xception实现
        layers = [
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ]

        # Entry flow
        layers += [
            XceptionBlock(64, 128, stride=2),
            XceptionBlock(128, 256, stride=2),
            XceptionBlock(256, 728, stride=2)
        ]

        # Middle flow
        for _ in range(16):
            layers.append(XceptionBlock(728, 728, skip_connection_type='sum'))

        # Exit flow
        layers += [
            XceptionBlock(728, 1024),
            XceptionBlock(1024, 2048, dilation=2)
        ]

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Backbone
        x = self.backbone(x)

        # ASPP
        x = self.aspp(x)

        # Decoder
        low_level_feat = self.backbone[4].conv3[0].pointwise  # 获取低层特征
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder(x)

        # Final upsampling
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return x

def dense_crf_wrapper(image_tensor, logits_tensor):
    image = image_tensor.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)[0]
    probs = logits_tensor.cpu().numpy()[0]

    d = densecrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
    U = -np.log(probs.reshape(2, -1))
    d.setUnaryEnergy(U.astype(np.float32))

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)

    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape(image.shape[:2])