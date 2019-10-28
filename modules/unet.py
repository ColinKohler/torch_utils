import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(UNet, self).__init__()
    self.input = DoubleConv(in_channels, 64)

    self.down1 = DownSample(64, 128)
    self.down2 = DownSample(128, 256)
    self.down3 = DownSample(256, 512)
    self.down4 = DownSample(512, 512)

    self.up1 = UpSample(1024, 256)
    self.up2 = UpSample(512, 128)
    self.up3 = UpSample(256, 64)
    self.up4 = UpSample(128, 64)

    self.out = nn.Conv2d(64, out_channels, 1)

  def forward(self, x):
    x1 = self.input(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)

    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    x = self.out(x)

    return x

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel=3, stride=2, pad=1):
    super(DoubleConv, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=pad),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel, stride=stride, padding=pad),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
    )

  def forward(self, x):
    x = self.conv(x)
    return x

class DownSample(nn.Module):
  def __init__(self, in_channels, out_channels, kernel=3, stride=0, pad=1):
    super(DownSample, self).__init__()
    self.conv = DoubleConv(in_channels, out_channels, kernel=kernel, stride=stride, pad=pad)
    self.pool = nn.MaxPool2d(2)

  def forward(self, x):
    x = self.conv(self.pool(x))
    return x

class UpSample(nn.Module):
  def __init__(self, in_channels, out_channels, kernel=2, stride=2):
    super(UpSample, self).__init__()
    self.conv_transpose = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel, stride=stride)
    self.conv = DoubleConv(in_channels, out_channels, kernel=3, stride=1, pad=1)

  def forward(self, x1, x2):
    x1 = self.conv_transpose(x1)

    # Pad the inputs so we can concat them together
    diff_y = x2.size(2) - x1.size(2)
    diff_x = x2.size(3) - x1.size(3)
    x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2))

    x = torch.cat([x2, x1], dim=1)
    x = self.conv(x)
    return x
