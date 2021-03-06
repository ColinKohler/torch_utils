import torch

def roundTensor(tensor, dec=0):
  return torch.round(tensor * 10**dec) / 10**dec

def argmax2d(tensor):
  n = tensor.size(0)
  d = tensor.size(-1)
  idx = tensor.view(n, -1).argmax(1)
  return torch.cat(((idx / d).view(-1, 1), (idx % d).view(-1, 1)), dim=1)

def topk2d(tensor, k=1):
  n = tensor.size(0)
  d = tensor.size(-1)
  val, idx = tensor.view(n, -1).topk(k)
  idx = torch.cat(((idx / d).view(-1, 1), (idx % d).view(-1, 1)), dim=1)
  return val, idx

def softUpdate(target_net, source_net, tau):
  for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
    target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def hardUpdate(target_net, source_net):
  target_net.load_state_dict(source_net.state_dict())
