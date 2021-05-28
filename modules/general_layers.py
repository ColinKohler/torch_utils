import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
  def init(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return x.view(x.size(0), -1)

class MaskedConv2d(nn.Conv2d):
  def __init__(self, mask_type, *args, **kwargs):
    super(MaskedConv2d, self).__init__(*args, **kwargs)
    assert mask_type in {'A', 'B'}
    self.register_buffer('mask', self.weight.data.clone())
    _, _, kH, kW = self.weight.size()
    self.mask.fill_(1)
    self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
    self.mask[:, :, kH // 2 + 1] = 0

  def forward(self, x):
    self.weight.data *= self.mask
    return super(MaskedConv2d, self).forward(x)

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1):
    super(ConvBlock, self).__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding, bias=False),
      nn.LeakyReLU(0.01, inplace=True),
    )

  def forward(self, x):
    out = self.conv(x)
    return out

class DoubleConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, bnorm=False):
    super(DoubleConvBlock, self).__init__()
    if bnorm:
      self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel, stride=1, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.01, inplace=True),
      )
    else:
      self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding, bias=False),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel, stride=1, padding=padding, bias=False),
        nn.LeakyReLU(0.01, inplace=True),
      )

    self.conv.apply(self.initWeights)

  def initWeights(self, m):
    if type(m) == nn.Conv2d:
      nn.init.kaiming_normal_(m.weight)

  def forward(self, x):
    out = self.conv(x)
    return out

class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1):
    super(ResBlock, self).__init__()

    self.conv = nn.Sequential(
      DoubleConvBlock(in_channels, out_channels, kernel, stride, padding),
      nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding, bias=False)
    )
    self.relu = nn.LeakyReLU(0.01, inplace=True)

  def forward(self, x):
    residual = x
    out = self.conv(x)
    out = out + residual
    out = self.relu(out)
    return out

class UpsamplingBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(UpsamplingBlock, self).__init__()

    self.conv = nn.Sequential(
      nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2, bias=False),
      DoubleConvBlock(in_channels, out_channels, kernel=3, stride=1, padding=1),
    )

  def forward(self, x):
    out = self.conv(x)
    return out

class ResUpsamplingBlock(nn.Module):
  def __init__(self, in_channels, out_channels, bnorm=False):
    super(ResUpsamplingBlock, self).__init__()
    self.conv = DoubleConvBlock(in_channels, out_channels, bnorm=bnorm)

  def forward(self, x1, x2):
    x1  = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=False)

    # Pad the inputs so we can concat them together
    diff_y = x2.size(2) - x1.size(2)
    diff_x = x2.size(3) - x1.size(3)
    x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2))

    x = torch.cat([x2, x1], dim=1)
    x = self.conv(x)
    return x
