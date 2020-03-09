# Adapted from https://github.com/fyu/drn/blob/master/drn.py

import math
import torch
import torch.nn as nn

def conv3x3(in_kernels, out_kernels, stride=1, padding=1, dilation=1):
  return nn.Conv2d(in_kernels, out_kernels, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_kernels, kernels, stride=1, downsample=None, dilation=(1,1), residual=True):
    super(BasicBlock, self).__init__()

    self.conv_block = nn.Sequential(
      conv3x3(in_kernels, kernels, stride, padding=dilation[0], dilation=dilation[0]),
      nn.BatchNorm2d(kernels),
      nn.ReLU(inplace=True),
      conv3x3(kernels, kernels, padding=dilation[1], dilation=dilation[1]),
      nn.BatchNorm2d(kernels),
    )

    self.relu = nn.ReLU(True)
    self.downsample = downsample
    self.stride = stride
    self.residual = residual

  def forward(self, x):
    residual = x
    out = self.conv_block(x)

    if self.downsample is not None:
      residual = self.downsample(x)
    if self.residual:
      out += residual
    out = self.relu(out)

    return out

class Bottleneck(nn.Module):
  expansion =  44

  def __init__(self, in_kernels, kernels, stride=1, downsample=None, dilation=(1,1), residual=True):
    super(Bottleneck, self).__init__()

    self.conv_block = nn.Sequential(
      nn.Conv2d(in_kernels, kernels, kernel_size=1, bias=False),
      nn.BatchNorm2d(kernels),
      nn.ReLU(inplace=True),
      nn.Conv2d(kernels, kernels, kernel_size=3, stride=stride, padding=dilation[1], bias=False, dilation=dilation[1]),
      nn.BatchNorm2d(kernels),
      nn.ReLU(inplace=True),
      nn.Conv2d(kernels, kernels*4, kernel_size=1, bias=False),
      nn.BatchNorm2d(kernels*4),
    )

    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride
    self.residual = residual

  def forward(self, x):
    residual = x

    out = self.conv_block(x)

    if self.downsample is not None:
      residual = self.downsample(x)
    if self.residual:
      out += residual
    out = self.relu(out)

    return out

class DRN(nn.Module):
  def __init__(self, in_channels, block, layers, num_classes,
                     channels=(16, 32, 64, 128, 256, 512, 512, 512),
                     out_map=False, out_middle=False, pool_size=28):
    super(DRN, self).__init__()

    self.in_channels = in_channels
    self.in_kernels = channels[0]
    self.out_map = out_map
    self.out_middle = out_middle
    self.out_dim = channels[-1]

    self.layer0 = nn.Sequential(
      nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
      nn.BatchNorm2d(channels[0]),
      nn.ReLU(inplace=True)
    )
    self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0], stride=1)
    self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], stride=2)
    self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
    self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
    self.layer5 = self._make_layer(block, channels[4], layers[4], dilation=2, new_level=False)
    self.layer6 = self._make_layer(block, channels[5], layers[5], dilation=4, new_level=False)
    self.layer7 = self._make_layer(BasicBlock, channels[6], layers[6], dilation=2, new_level=False, residual=False)
    self.layer8 = self._make_layer(BasicBlock, channels[7], layers[7], dilation=1, new_level=False, residual=False)

    if num_classes > 0:
      self.avgpool = nn.AvgPool2d(pool_size)
      self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, kernels, blocks, stride=1, dilation=1, new_level=True, residual=True):
    assert dilation == 1 or dilation % 2 == 0

    downsample = None
    if stride != 1 or self.in_kernels != kernels * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.in_kernels, kernels * block.expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(kernels * block.expansion)
      )

    layers = list()
    first_dilation = (1,1) if dilation == 1 else (dilation // 2 if new_level else dilation, dilation)
    layers.append(block(self.in_kernels, kernels, stride, downsample, dilation=first_dilation, residual=residual))
    self.in_kernels = kernels * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.in_kernels, kernels, residual=residual, dilation=(dilation, dilation)))

    return nn.Sequential(*layers)

  def forward(self, x):
    y = list()

    x = self.layer0(x)

    x = self.layer1(x)
    y.append(x)
    x = self.layer2(x)
    y.append(x)

    x = self.layer3(x)
    y.append(x)

    x = self.layer4(x)
    y.append(x)

    x = self.layer5(x)
    y.append(x)

    x = self.layer6(x)
    y.append(x)

    x = self.layer7(x)
    y.append(x)

    x = self.layer8(x)
    y.append(x)

    return x

    # if self.out_map:
    #   x = self.fc(x)
    # else:
    #   x = self.avgpool(x)
    #   self.fc(x)
    #   x = x.view(x.size(0), -1)

    # if self.out_middle:
    #   return x, y
    # else:
    #   return x

def drn_c_26(in_channels, num_classes):
  return DRN(in_channels, BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], num_classes, out_map=True, out_middle=False)
