import torch

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
