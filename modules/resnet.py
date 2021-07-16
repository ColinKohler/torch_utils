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
                 base_width=64, dilation=1, norm_layer=None):
      super(BasicBlock, self).__init__()
      if norm_layer is None:
        norm_layer = nn.BatchNorm2d

      self.conv1 = conv3x3(in_kernels, kernels, stride)
      self.bn1 = norm_layer(kernels)
      self.conv2 = conv3x3(kernels, kernels)
      self.bn2 = norm_layer(kernels)

      self.downsample = downsample
      self.stride = stride
      self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
      identity = x

      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu(out)

      out = self.conv2(out)
      out = self.bn2(out)

      if self.downsample is not None:
        identity = self.downsample(x)

      out += identity
      out = self.relu(out)

      return out

class BottleneckBlock(nn.Module):
  expansion = 1

  def __init__(self, in_kernels, kernels, stride=1, downsample=None, groups=1,
               base_width=64, dilation=1, norm_layer=None):
      super(BottleneckBlock, self).__init__()
      if norm_layer is None:
        norm_layer = nn.BatchNorm2d

      width = int(kernels * (base_width / 64.)) * groups

      self.conv1 = conv1x1(in_kernels, width)
      #self.bn1 = norm_layer(width)
      self.conv2 = conv3x3(width, width, stride, groups, dilation)
      #self.bn2 = norm_layer(width)
      self.conv3 = conv1x1(width, kernels * self.expansion)
      #self.bn3 = norm_layer(kernels * self.expansion)

      self.downsample = downsample
      self.stride = stride
      self.relu = nn.LeakyReLU(0.1, inplace=True)

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
