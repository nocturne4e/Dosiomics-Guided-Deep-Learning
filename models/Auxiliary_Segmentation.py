import torch
import torch.nn as nn

class UpsampleConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleConvBNReLU, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Auxiliary_Net(nn.Module):
    def __init__(self):
        super(Auxiliary_Net, self).__init__()
    
        self.stage1 = UpsampleConvBNReLU(512, 256)
        
        self.stage2 = UpsampleConvBNReLU(256, 128)

        self.stage3 = UpsampleConvBNReLU(128, 64)

        self.stage4 = UpsampleConvBNReLU(64, 1)
        

    def forward(self, x):
        
        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        return x

