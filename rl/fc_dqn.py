import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
  def __init__(self, state_dim, num_actions):

