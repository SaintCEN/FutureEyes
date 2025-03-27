import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)
        total_dice = 0
        for c in range(y_pred.shape[1]):
            intersection = torch.sum(y_true[:, c] * y_pred[:, c], dim=(1, 2))
            union = torch.sum(y_true[:, c], dim=(1, 2)) + torch.sum(y_pred[:, c], dim=(1, 2))
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            total_dice += dice.mean()  # 按类别平均后累加
        return 1 - total_dice / y_pred.shape[1]  # 再取类别平均

class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, AB_layer=False):
        super(ConvolutionBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.AB_layer = AB_layer
        if AB_layer:
            self.attention = Attention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.AB_layer:
            x = self.attention(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, depth=4, base_channels=16, dropout=0.3, AB_block=True, fixed_filter=False):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for i in range(depth):
            out_channels = base_channels if fixed_filter else base_channels * (2 ** i)
            self.layers.append(nn.Sequential(
                ConvolutionBlock(current_channels, out_channels, AB_layer=AB_block),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout)
            ))
            current_channels = out_channels
        self.out_channels = current_channels

    def forward(self, x):
        skips = []
        for layer in self.layers:
            x = layer[0](x)
            skips.append(x)
            x = layer[1](x)
            x = layer[2](x)
        return x, skips

class MidBlock(nn.Module):
    def __init__(self, in_channels, depth=4, reduction=2, segmentation=False):
        super(MidBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.segmentation = segmentation
        current_channels = in_channels
        f_reduction = 1

        for _ in range(depth):
            out_channels = in_channels // f_reduction
            self.layers.append(nn.Sequential(
                nn.Conv2d(current_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                Attention(out_channels)
            ))
            current_channels = out_channels
            f_reduction *= reduction

        if segmentation:
            self.seg_conv = nn.Sequential(
                nn.Conv2d(current_channels, in_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        outputs = []
        xe1 = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0:
                xe1 = x
            outputs.append(x)

        if self.segmentation:
            x = self.seg_conv(x)
            x = x + xe1
            outputs.append(x)
        return outputs

class Decoder(nn.Module):
    def __init__(self, in_channels, depth=4, base_channels=16, dropout=0.3, AB_block=True):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ConvolutionBlock(base_channels * 2, base_channels, AB_layer=AB_block),
                nn.Dropout2d(dropout)
            ))

    def forward(self, x, skips):
        for i, layer in enumerate(self.layers):
            x = layer[0](x)
            if x.size()[2:] != skips[-(i + 1)].size()[2:]:
                x = F.interpolate(x, size=skips[-(i + 1)].size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skips[-(i + 1)]], dim=1)
            x = layer[1](x)
            x = layer[2](x)
        return x

class LWBNA_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, depth=4, base_channels=128, dropout=0.3, AB_block=True):
        super(LWBNA_UNet, self).__init__()
        self.encoder = Encoder(in_channels, depth, base_channels, dropout, AB_block, fixed_filter=True)
        self.mid = MidBlock(self.encoder.out_channels, depth, segmentation=True)
        self.bottleneck_conv = ConvolutionBlock(base_channels, base_channels, AB_layer=AB_block)
        self.decoder = Decoder(base_channels, depth, base_channels, dropout, AB_block)
        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x, skips = self.encoder(x)
        mid_outputs = self.mid(x)
        x = self.bottleneck_conv(mid_outputs[-1])
        x = self.decoder(x, skips)
        x = self.final_conv(x)
        return x