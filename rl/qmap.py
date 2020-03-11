import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils.modules.general_layers import DoubleConvBlock, ResUpsamplingBlock, Flatten

import ipdb

class QMap(nn.Module):
  def __init__(self, in_channels, out_channels, device):
    super(ValueModel, self).__init__()
    self.device = device

    self.input = DoubleConvBlock(in_channels, 8)
    self.down1 = DoubleConvBlock(8, 16, stride=2)
    self.down2 = DoubleConvBlock(16, 32, stride=2)
    self.down3 = DoubleConvBlock(32, 64, stride=2)
    self.down4 = DoubleConvBlock(64, 64, stride=2)
    self.forward_conv = nn.Conv2d(64, 64, 1)
    self.up1 = ResUpsamplingBlock(128, 64)
    self.up2 = ResUpsamplingBlock(96, 32)
    self.up3 = ResUpsamplingBlock(48, 16)
    self.up4 = ResUpsamplingBlock(24, 8)
    self.out = nn.Conv2d(8, out_channels, 1, padding=0)

  def forward(self, x):
    x_1 = self.input(x)
    x_2 = self.down1(x_1)
    x_3 = self.down2(x_2)
    x_4 = self.down3(x_3)
    x_5 = self.down4(x_4)

    q_values = self.forward_conv(x_5)
    q_values = self.up1(q_values, x_4)
    q_values = self.up2(q_values, x_3)
    q_values = self.up3(q_values, x_2)
    q_values = self.up4(q_values, x_1)
    q_values = self.out(q_values)

    return q_values
