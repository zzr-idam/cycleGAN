import torch
import torch.nn as nn
import torch.nn.functional as F

class  ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            if down
            else #nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)  if use_act else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
            nn.PReLU()
        )
    
    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.residual = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        return x + self.residual(x)


class Generator(nn.Module):
    def __init__(self, img_channels=3, features=64, residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, features, 3, 1, 1, padding_mode="reflect"),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        )

        self.down_blocks = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 256, kernel_size= 3, stride=1, padding=1),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(residuals)]
        )

        self.up_blocks = nn.Sequential(
            ConvBlock(256, 128, down=False, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 64, down=False, kernel_size=3, stride=1, padding=1),
        )

        self.last = nn.Conv2d(features, img_channels, 7, 1, 3, padding_mode="reflect")
        self.smooth_down = nn.Sequential(
          nn.Conv2d(64, 128, 1, 1, 0),
          nn.PReLU(),
          nn.Conv2d(128, 256, 1, 1, 0),
          nn.PReLU()
        )
        self.smooth_up = nn.Sequential(
          nn.Conv2d(256, 128, 1, 1, 0),
          nn.PReLU(),
          nn.Conv2d(128, 64, 1, 1, 0),
          nn.PReLU()
        )

    def forward(self, x):
        f = F.interpolate(x, (256, 256),
                            mode='bilinear', align_corners=True)
        f = self.initial(f)
        f = F.interpolate(f, (64, 64),
                            mode='bilinear', align_corners=True)
        f = self.smooth_down(f)
        f = self.res_blocks(f)
        f = F.interpolate(f, (256, 256),
                            mode='bilinear', align_corners=True)
        f = self.smooth_up(f)
        f = self.last(f)
        f = F.interpolate(f, (x.shape[2], x.shape[3]),
                            mode='bilinear', align_corners=True)
        return torch.tanh(f + x)