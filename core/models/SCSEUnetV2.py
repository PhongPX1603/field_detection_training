import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models



# SCSE module
class SCSE(nn.Module):
    def __init__(self, in_ch):
        super(SCSE, self).__init__()
        self.spatial_gate = SpatialGate2d(in_ch, 16)  # 16
        self.channel_gate = ChannelGate2d(in_ch)

    def forward(self, x):
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1 + g2  # x = g1*x + g2*x
        return x


# Space Gating
class SpatialGate2d(nn.Module):
    def __init__(self, in_ch, r=16):
        super(SpatialGate2d, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)
        x = input_x * x

        return x


# Channel Gating
class ChannelGate2d(nn.Module):
    def __init__(self, in_ch):
        super(ChannelGate2d, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = input_x * x

        return x


# Encoding continuous convolutional layer
def contracting_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(kernel_size=(3, 3), in_channels=out_channels, out_channels=out_channels, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block


# Decode upsampling convolutional layer
class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(3, 3), stride=2, padding=1,
                                     output_padding=1, dilation=1)
        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.spa_cha_gate = SCSE(out_channels)

    def forward(self, d, e=None):
        d = self.up(d)
        # d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        # concat
        if e is not None:
            cat = torch.cat([e, d], dim=1)
            out = self.block(cat)
        else:
            out = self.block(d)
        out = self.spa_cha_gate(out)
        return out


# Output layer
def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(1, 1), in_channels=in_channels, out_channels=out_channels),
        nn.UpsamplingBilinear2d(scale_factor=4)
    )
    return block




def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU()
    )

from .model import Model
# SCSE U-Net
class SCSEUnet(Model):
    def __init__(self, num_classes=2, pretrained_backbone=False):
        super(SCSEUnet, self).__init__()

        self.base_model = models.resnet34(pretrained=pretrained_backbone)


        self.base_layers = list(self.base_model.children())
        self.conv_encode1 = nn.Sequential(
            *self.base_layers[:3],
            convrelu(64, 64, 1, 0),
            *self.base_layers[3:5],
            convrelu(64, 64, 1, 0),
            SCSE(64)
        )
        self.conv_encode2 = nn.Sequential(
            self.base_layers[5],
            convrelu(128, 128, 1, 0),
            SCSE(128)
        )
        self.conv_encode3 = nn.Sequential(
            self.base_layers[6],
            convrelu(256, 256, 1, 0),
            SCSE(256)
        )
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            self.base_layers[7],
            convrelu(512, 512, 1, 0),
            SCSE(512)
        )
        # Decode
        self.conv_decode3 = expansive_block(512, 256, 256)
        self.conv_decode2 = expansive_block(256, 128, 128)
        self.conv_decode1 = expansive_block(128, 64, 64)
        self.final_layer = final_block(64, num_classes)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_block2 = self.conv_encode2(encode_block1)
        encode_block3 = self.conv_encode3(encode_block2)
        bottleneck = self.bottleneck(encode_block3)
        # Decode
        decode_block3 = self.conv_decode3(bottleneck, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)

        final_layer = self.final_layer(decode_block1)
        out = nn.Softmax(1)(final_layer)  # Can be annotated, according to the situation

        return out
