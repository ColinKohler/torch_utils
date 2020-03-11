import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_utils.utils as torch_utils
from conv_dqn import ConvDQN
from fc_dqn import FcDQN

class DDQN(nn.Module):
  def __init__(self, state_shape, num_actions, conv_filters=[8, 16, 32, 64], tau=1e-2):
    if len(state_shape) == 1:
      self.model = FcDQN(state_shape, num_actions)
      self.target_model = FcDQN(state_shape, num_actions)
    else:
      self.model = ConvDQN(state_shape, num_actions, conv_filters)
      self.target_model = ConvDQN(state_shape, num_actions, conv_filters)

    self.tau = tau
    torch_utils.hardUpdate(target_model, model)

  def forward(self, x):
    return self.model(x)

  def targetForward(self, x):
    return self.target_model(x)

  def updateTarget(self):
    torch_utils.softUpdate(target_model, model, self.tau)
