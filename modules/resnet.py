import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_kernels, out_kernels, stride=1):
  return nn.Conv2d(in_kernels, out_kernels, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_kernels, out_kernels, stride=1, groups=1, dilation=1):
  return nn.Conv2d(in_kernels, out_kernels, kernel_size=3, stride=stride,
                   padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_kernels, kernels, stride=1, downsample=None, groups=1,
                 dilation=1, norm_layer=None):
      super(BasicBlock, self).__init__()
      if norm_layer is None:
        norm_layer = nn.BatchNorm2d

      self.conv1 = conv3x3(in_kernels, kernels, stride)
      #self.bn1 = norm_layer(kernels)
      self.conv2 = conv3x3(kernels, kernels)
      #self.bn2 = norm_layer(kernels)

      self.downsample = downsample
      self.stride = stride
      self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
      identity = x

      out = self.conv1(x)
      #out = self.bn1(out)
      out = self.relu(out)

      out = self.conv2(out)
      #out = self.bn2(out)

      if self.downsample is not None:
        identity = self.downsample(x)

      out += identity
      out = self.relu(out)

      return out

class BottleneckBlock(nn.Module):
  expansion = 2

  def __init__(self, in_kernels, kernels, stride=1, downsample=None, groups=1,
                     dilation=1, norm_layer=None):
      super(BottleneckBlock, self).__init__()
      if norm_layer is None:
        norm_layer = nn.BatchNorm2d

      self.conv1 = conv1x1(in_kernels, kernels)
      #self.bn1 = norm_layer(kernels)
      self.conv2 = conv3x3(kernels, kernels, stride, groups, dilation)
      #self.bn2 = norm_layer(kernels)
      self.conv3 = conv1x1(kernels, kernels * self.expansion)
      #self.bn3 = norm_layer(kernels * self.expansion)

      self.downsample = downsample
      self.stride = stride
      self.relu = nn.LeakyReLU(0.01, inplace=True)

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    #out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    #out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    #out = self.bn3(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out

class UpsamplingBlock(nn.Module):
  def __init__(self, in_channels_1, in_channels_2, out_channels):
    super(UpsamplingBlock, self).__init__()
    self.upsample_conv = nn.Conv2d(in_channels_1, in_channels_1, 3, stride=1, padding=1, bias=True)
    self.relu = nn.LeakyReLU(0.01, inplace=True)
    self.conv1 = nn.Conv2d(in_channels_1 + in_channels_2, out_channels, 1, bias=True)

  def forward(self, x1, x2):
    x1  = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=False)

    # Pad the inputs so we can concat them together
    diff_y = x2.size(2) - x1.size(2)
    diff_x = x2.size(3) - x1.size(3)
    x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2))

    x1 = self.upsample_conv(x1) + x1
    x1 = self.relu(x1)

    x = torch.cat([x2, x1], dim=1)
    out = self.conv1(x)

    return out

class BasicUpsamplingBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(BasicUpsamplingBlock, self).__init__()
    self.conv1 = BasicBlock(in_channels, int(in_channels/2), downsample=nn.Conv2d(in_channels, int(in_channels/2), 1, bias=False))
    self.conv2 = BasicBlock(int(in_channels/2), out_channels, downsample=nn.Conv2d(int(in_channels/2), out_channels, 1, bias=False))

  def forward(self, x1, x2):
    x1  = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=False)

    # Pad the inputs so we can concat them together
    diff_y = x2.size(2) - x1.size(2)
    diff_x = x2.size(3) - x1.size(3)
    x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2))

    x = torch.cat([x2, x1], dim=1)
    out = self.conv1(x)
    out = self.conv2(out)

    return out

class BottleneckUpsamplingBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(BottleneckUpsamplingBlock, self).__init__()
    self.conv1 = BottleneckBlock(in_channels, int(in_channels/2), downsample=nn.Conv2d(in_channels, int(in_channels/2), 1, bias=False))
    self.conv2 = BottleneckBlock(int(in_channels/2), out_channels, downsample=nn.Conv2d(int(in_channels/2), out_channels, 1, bias=False))

  def forward(self, x1, x2):
    x1  = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=False)

    # Pad the inputs so we can concat them together
    diff_y = x2.size(2) - x1.size(2)
    diff_x = x2.size(3) - x1.size(3)
    x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2))

    x = torch.cat([x2, x1], dim=1)
    out = self.conv1(x)
    out = self.conv2(out)

    return out
