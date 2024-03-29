import torch

def convert1h(x, n):
  b = x.size(0)

  x_1h = torch.FloatTensor(b, n).to(x.device)
  x_1h.zero_()
  x_1h.scatter_(1, x.long(), 1)

  return x_1h

def normalizeTensor(tensor):
  if len(torch.unique(tensor)) == 1:
    return tensor
  return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))

def roundTensor(tensor, dec=0):
  return torch.round(tensor * 10**dec) / 10**dec

def argmax2d(tensor):
  n = tensor.size(0)
  d = tensor.size(-1)
  idx = tensor.view(n, -1).argmax(1)
  return torch.cat((torch.divide(idx, d, rounding_mode='trunc').view(-1, 1), (idx % d).view(-1, 1)), dim=1)

def argmax3d(tensor):
  n = tensor.size(0)
  c = tensor.size(1)
  d = tensor.size(2)
  idx = tensor.contiguous().view(n, -1).argmax(1)
  return torch.cat(((idx / (d**2)).view(-1, 1),
                    ((idx % (d**2)) / d).view(-1, 1),
                    ((idx % (d**2)) % d).view(-1, 1)), dim=1)

def argmax4d(tensor):
  n = tensor.size(0)
  c1 = tensor.size(1)
  c2 = tensor.size(2)
  d = tensor.size(3)
  idx = tensor.contiguous().view(n, -1).argmax(1)
  return torch.cat((torch.divide(idx, d**2 * c2, rounding_mode='trunc').view(-1, 1),
                    (idx % (d**2 *c2) / d**2).view(-1, 1),
                    (((idx % (d**2 * c2)) % d**2) / d).view(-1, 1),
                    (((idx % (d**2 * c2)) % d**2) % d).view(-1, 1)), dim=1)

def topk2d(tensor, k=1):
  n = tensor.size(0)
  d = tensor.size(-1)
  val, idx = tensor.view(n, -1).topk(k, dim=1)
  idx = torch.cat((torch.divide(idx, d, rounding_mode='trunc').view(-1, 1), (idx % d).view(-1, 1)), dim=1)
  return val, idx

def topk3d(tensor, k=1):
  n = tensor.size(0)
  c = tensor.size(1)
  d = tensor.size(2)
  idx = tensor.contiguous().view(n, -1).topk(k, dim=1)
  return torch.cat((torch.divide(idx, d**2, rounding_mode='trunc').view(-1, 1),
                    (idx % d**2 / d).view(-1, 1),
                    (idx % d**2 / d).view(-1, 1)), dim=1)

def argmax4d(tensor):
  n = tensor.size(0)
  c1 = tensor.size(1)
  c2 = tensor.size(2)
  d = tensor.size(3)
  idx = tensor.contiguous().view(n, -1).topk(k, dim=1)
  return torch.cat((torch.divide(idx, d**2 * c2, rounding_mode='trunc').view(-1, 1),
                    (idx % (d**2 *c2) / d**2).view(-1, 1),
                    (((idx % (d**2 * c2)) % d**2) / d).view(-1, 1),
                    (((idx % (d**2 * c2)) % d**2) % d).view(-1, 1)), dim=1)


def sample2d(tensor, k=1):
  if tensor.dim() == 2:
    n = 1
    d = tensor.size(-1)
  else:
    n = tensor.size(0)
    d = tensor.size(-1)
  idx =  torch.multinomial(tensor.reshape(n, -1), k)
  return torch.cat((torch.divide(idx, d, rounding_mode='trunc').view(-1, 1), (idx % d).view(-1, 1)), dim=1)

def softUpdate(target_net, source_net, tau):
  for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
    target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def hardUpdate(target_net, source_net):
  target_net.load_state_dict(source_net.state_dict())

def dictToCpu(dictionary):
  cpu_dict = {}
  for key, value in dictionary.items():
    if isinstance(value, torch.Tensor):
      cpu_dict[key] = value.cpu()
    elif isinstance(value, dict):
      cpu_dict[key] = dictToCpu(value)
    else:
      cpu_dict[key] = value
  return cpu_dict
