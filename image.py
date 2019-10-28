import torch

def buildGrid(source_size, target_size):
  k = float(target_size) / float(source_size)
  direct = torch.linspace(0, k, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
  full = torch.cat([direct, direct.transpose(1,0)], dim=2).unsqueeze(0)

  return full
