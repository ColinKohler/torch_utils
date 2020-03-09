import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils.modules.general_layers import DoubleConvBlock, Flatten

class ConvDQN(nn.Module):
  def __init__(self, state_shape, num_actions, conv_filters=[8, 16, 32, 64]):
    super(ConvDQN, self).__init__()

    state_dim = state_shape[0]
    state_size = state_shape[-1]
    post_conv_size = state_size / (2*num_conv_layers)
    conv_filters = [state_dim] + conv_filters
    num_conv_layers = len(conv_filters) - 1

    conv_layers = list()
    for i in range(num_conv_layers):
      conv_layers.append(DoubleConvBlock(conv_filters[i], conv_filters[i+1], stride=2))

    self.model = nn.Sequential(
      conv_layers + [
        Flatten(),
        nn.Linear(post_conv_size, 256),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Linear(256, 128),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Linear(128, 64),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Linear(64, num_actions)
      ]
    )

  def forward(self, x):
    return self.model(x)
