import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_kernels, out_kernels, stride=1):
  return nn.Conv2d(in_kernels, out_kernels, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_kernels, out_kernels, stride=1, groups=1, dilation=1):
  return nn.Conv2d(in_kernels, out_kernels, kernel_size=3, stride=stride,
                   padding=dilation, groups=groups, bias=False, dilation=dilation)

def getIncomingShape(incoming):
  size = incoming.size()
  return [size[0], size[1], size[2], size[3]]

def interleave(tensors, axis):
  old_shape = getIncomingShape(tensors[0])[1:]
  new_shape = [-1] + old_shape

  new_shape[axis] *= len(tensors)
  stacked = torch.stack(tensors, axis+1)
  reshaped = stacked.view(new_shape)

  return reshaped

def makeLayer(block, in_kernels, kernels, blocks, stride=1):
  downsample = None
  if stride != 1 or in_kernels != kernels * block.expansion:
    downsample = nn.Sequential(
      nn.Conv2d(in_kernels, kernels * block.expansion, kernel_size=1, stride=stride, bias=False)
    )

  layers = list()
  layers.append(block(in_kernels, kernels, stride, downsample))
  in_kernels = kernels * block.expansion
  for i in range(1, blocks):
    layers.append(block(in_kernels, kernels))

  return nn.Sequential(*layers)

class Basic1x1Block(nn.Module):
    expansion = 1

    def __init__(self, in_kernels, kernels, stride=1, downsample=None, norm_layer=None):
      super(Basic1x1Block, self).__init__()
      if norm_layer is None:
        norm_layer = nn.BatchNorm2d

      self.conv1 = conv1x1(in_kernels, kernels, stride)
      self.bn1 = norm_layer(kernels)
      self.conv2 = conv1x1(kernels, kernels, stride)
      self.bn2 = norm_layer(kernels)

      self.downsample = downsample
      self.stride = stride
      self.relu = nn.LeakyReLU(0.01, inplace=True)

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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_kernels, kernels, stride=1, downsample=None, groups=1,
                 dilation=1, norm_layer=None):
      super(BasicBlock, self).__init__()
      if norm_layer is None:
        norm_layer = nn.BatchNorm2d

      self.conv1 = conv3x3(in_kernels, kernels, stride)
      self.bn1 = norm_layer(kernels)
      self.conv2 = conv3x3(kernels, kernels)
      self.bn2 = norm_layer(kernels)

      self.downsample = downsample
      self.stride = stride
      self.relu = nn.LeakyReLU(0.01, inplace=True)

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
  expansion = 2

  def __init__(self, in_kernels, kernels, stride=1, downsample=None, groups=1,
                     dilation=1, norm_layer=None):
      super(BottleneckBlock, self).__init__()
      if norm_layer is None:
        norm_layer = nn.BatchNorm2d

      self.conv1 = conv1x1(in_kernels, kernels)
      self.bn1 = norm_layer(kernels)
      self.conv2 = conv3x3(kernels, kernels, stride, groups, dilation)
      self.bn2 = norm_layer(kernels)
      self.conv3 = conv1x1(kernels, kernels * self.expansion)
      self.bn3 = norm_layer(kernels * self.expansion)

      self.downsample = downsample
      self.stride = stride
      self.relu = nn.LeakyReLU(0.01, inplace=True)

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out

#class UpsamplingBlock(nn.Module):
#  def __init__(self, in_channels_1, in_channels_2, out_channels):
#    super(UpsamplingBlock, self).__init__()
#    self.upsample_conv = nn.Conv2d(in_channels_1, in_channels_1, 3, stride=1, padding=1, bias=True)
#    self.relu = nn.LeakyReLU(0.01, inplace=True)
#    self.conv1 = nn.Conv2d(in_channels_1 + in_channels_2, out_channels, 1, bias=True)
#
#  def forward(self, x1, x2):
#    x1  = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=False)
#
#    # Pad the inputs so we can concat them together
#    diff_y = x2.size(2) - x1.size(2)
#    diff_x = x2.size(3) - x1.size(3)
#    x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2,
#                    diff_y // 2, diff_y - diff_y // 2))
#
#    x1 = self.upsample_conv(x1) + x1
#    x1 = self.relu(x1)
#
#    x = torch.cat([x2, x1], dim=1)
#    out = self.conv1(x)
#
#    return out

class UnpoolingAsConvolution(nn.Module):
  def __init__(self, in_kernels, out_kernels):
    super(UnpoolingAsConvolution, self).__init__()

    self.conv_A = nn.Conv2d(in_kernels, out_kernels, kernel_size=(3,3), stride=1, padding=1)
    self.conv_B = nn.Conv2d(in_kernels, out_kernels, kernel_size=(2,3), stride=1, padding=0)
    self.conv_C = nn.Conv2d(in_kernels, out_kernels, kernel_size=(3,2), stride=1, padding=0)
    self.conv_D = nn.Conv2d(in_kernels, out_kernels, kernel_size=(2,2), stride=1, padding=0)

  def forward(self, x):
    out_a = self.conv_A(x)

    padded_b = F.pad(x, (1, 1, 0, 1))
    out_b = self.conv_B(padded_b)

    padded_c = F.pad(x, (0, 1, 1, 1))
    out_c = self.conv_C(padded_c)

    padded_d = F.pad(x, (0, 1, 0, 1))
    out_d = self.conv_D(padded_d)

    out_left = interleave([out_a, out_b], axis=2)
    out_right = interleave([out_c, out_d], axis=2)
    out = interleave([out_left, out_right], axis=3)

    return out

class UpsamplingBlock(nn.Module):
  def __init__(self, in_kernels, kernels):
    super(UpsamplingBlock, self).__init__()

    self.layer = nn.Sequential(
      UnpoolingAsConvolution(in_kernels, kernels),
      nn.BatchNorm2d(kernels),
      nn.LeakyReLU(0.01, inplace=False),
      nn.Conv2d(kernels, kernels, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(kernels)
    )

    self.res_layer = nn.Sequential(
      UnpoolingAsConvolution(in_kernels, kernels),
      nn.BatchNorm2d(kernels)
    )
    self.relu = nn.LeakyReLU(0.01, inplace=False)

  def forward(self, x):
    identity = x

    x = self.layer(x)
    identity = self.res_layer(identity)

    x += identity
    x = self.relu(x)

    return x

class CatConv(nn.Module):
  def __init__(self, in_kernels_1, in_kernels_2, kernels):
    super(CatConv, self).__init__()

    self.conv = nn.Conv2d(in_kernels_1 + in_kernels_2, kernels, kernel_size=1, bias=True)

  def forward(self, x1, x2):
    x1 = torch.cat([x2, x1], dim=1)
    x1 = self.conv(x1)

    return x1
