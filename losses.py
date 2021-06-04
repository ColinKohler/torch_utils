import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
  def __init__(self, device, alpha=None, gamma=2, smooth=1e-5, size_average=True):
    super(FocalLoss, self).__init__()
    self.device = device
    self.alpha = alpha
    self.gamma = gamma
    self.smooth = smooth
    self.size_average = size_average

    if self.smooth is not None:
      if self.smooth < 0 or self.smooth > 1.0:
        raise ValueError('Smooth value should be in [0,1]')

  def forward(self, logit, target, alpha=None):
    N = logit.size(0)
    C = logit.size(1)

    if logit.dim() > 2:
      # N, C, d1, d2, ..., dn --> N * d1 * d2 * ... * dn, C
      logit = logit.view(N, C, -1)
      logit = logit.permute(0, 2, 1).contiguous()
      logit = logit.view(-1, C)

    # N, d1, d2, ..., dn --> N * d1 * d2 * ... * dn, 1
    target = torch.squeeze(target, 1)
    target = target.view(-1, 1)

    idx = target.cpu().long()
    one_hot_key = torch.FloatTensor(target.size(0), C).zero_()
    one_hot_key = one_hot_key.scatter_(1, idx, 1)
    one_hot_key = torch.clamp(one_hot_key,
                              self.smooth / (C - 1),
                              1.0 - self.smooth)
    one_hot_key = one_hot_key.to(self.device)

    pt = (one_hot_key * logit).sum(1) + self.smooth
    logpt = pt.log()

    if alpha is None:
      alpha = self.alpha.to(self.device)
    alpha = alpha[idx].squeeze()
    loss = -1 * alpha * torch.pow((1 - pt), self.gamma) * logpt
    loss = loss.view(N, -1)

    if self.size_average:
      loss = loss.mean(1)
    else:
      loss = loss.sum(1)

    return loss

def scalarToSupport(x, support_size, eps=1):
  '''
  Transform a scalar to a categorical representation.
  Number of categories = 2 * support_size + 1
  '''
  # Reduce the scale (https://arxiv.org/abs/1805.11593)
  x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1 + eps * x)

  # Encode vector
  x = torch.clamp(x, -support_size, support_size)
  floor = x.floor()
  prob = x - floor

  logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
  logits.scatter_(2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1))

  idxs = floor + support_size + 1
  prob = prob.masked_fill_(2 * support_size < idxs, 0.0)
  idxs = idxs.masked_fill_(2 * support_size < idxs, 0.0)
  logits.scatter_(2, idxs.long().unsqueeze(-1), prob.unsqueeze(-1))

  return logits

def supportToScalar(logits, support_size, eps=1):
  '''
  Transform a categorical representation to a scalar.
  '''
  probabilities = torch.softmax(logits, dim=1)
  support = (
    torch.tensor([x for x in range(-support_size, support_size + 1)])
    .expand(probabilities.shape)
    .float()
    .to(probabilities.device)
  )
  x = torch.sum(support * probabilities, dim=1, keepdim=True)

  # Invert the scaling (https://arxiv.org/abs/1805.11593)
  x = torch.sign(x) * (
    ((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps))
    **2
    - 1
  )

  return x
