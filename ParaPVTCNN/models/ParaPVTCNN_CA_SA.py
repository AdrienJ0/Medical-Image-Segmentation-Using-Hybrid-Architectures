import torch
from torchvision import models as resnet_model
from torch import nn
import timm
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.mul(x, y)
        return y
    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

### Spacial attention block

""" class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x """

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(in_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class ParaPVTCNN_CA_SA(nn.Module):
    def __init__(self, n_channels=3, num_classes=9, dim=320, patch_size=2):
        super(ParaPVTCNN_CA_SA, self).__init__()
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.dim = dim
        embed_dim = [64, 128, 256, 512]  # PVT uses different dimensions for different stages

        resnet = resnet_model.resnet34(weights=resnet_model.ResNet34_Weights.DEFAULT)

        # Create the PVT model
        self.transformer = timm.create_model(
            'pvt_v2_b2_li',
            pretrained=True,
            features_only=True,
        )

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Update SEBlock initialization with correct input channels
        self.SE_1 = SEBlock(embed_dim[3] + 512)
        self.SE_2 = SEBlock(dim + 256)
        self.SE_3 = SEBlock(embed_dim[1] + 128)
        
        self.SA = SpatialAttention()

        self.decoder1 = DecoderBottleneckLayer(embed_dim[3] + 512)
        self.decoder2 = DecoderBottleneckLayer(dim + embed_dim[2] + 512) # Trouver la bonne formule mathématique
        self.decoder3 = DecoderBottleneckLayer(512)

        self.up3_1 = nn.ConvTranspose2d(embed_dim[3] + 512, embed_dim[2] + 256, kernel_size=2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(dim + embed_dim[2] + 512, embed_dim[1] + 128, kernel_size=2, stride=2)
        self.up1_1 = nn.ConvTranspose2d(512, embed_dim[0], kernel_size=4, stride=4)
        self.out = nn.Conv2d(embed_dim[0], num_classes, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Get features from PVT model
        features = self.transformer(x)
        v1_cnn = features[0]
        v2_cnn = features[1]
        v3_cnn = features[2]
        v4_cnn = features[3]

        # Ensure the dimensions match
        v4_cnn = nn.functional.interpolate(v4_cnn, size=e4.shape[2:], mode='bilinear', align_corners=False)
        cat_1 = torch.cat([v4_cnn, e4], dim=1)
        cat_1 = self.SE_1(cat_1)
        cat_1 = self.SA(cat_1) * cat_1
        cat_1 = self.decoder1(cat_1)
        cat_1 = self.up3_1(cat_1)

        v3_cnn = nn.functional.interpolate(v3_cnn, size=e3.shape[2:], mode='bilinear', align_corners=False)
        cat_2 = torch.cat([v3_cnn, e3], dim=1)
        cat_2 = self.SE_2(cat_2)
        cat_2 = self.SA(cat_2) * cat_2
        cat_2 = torch.cat([cat_2, cat_1], dim=1)
        cat_2 = self.decoder2(cat_2)
        cat_2 = self.up2_1(cat_2)

        v2_cnn = nn.functional.interpolate(v2_cnn, size=e2.shape[2:], mode='bilinear', align_corners=False)
        cat_3 = torch.cat([v2_cnn, e2], dim=1)
        cat_3 = self.SE_3(cat_3)
        cat_3 = self.SA(cat_3) * cat_3
        cat_3 = torch.cat([cat_3, cat_2], dim=1)
        cat_3 = self.decoder3(cat_3)
        cat_3 = self.up1_1(cat_3)

        out = self.out(cat_3)

        return out
