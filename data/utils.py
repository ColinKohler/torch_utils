import numpy as np
import numpy.random as npr
import torch
from torch.utils import data

def torchDataset(dataset, dataset_cls, batch_size, device):
  dataset = list(map(np.array, zip(*dataset)))
  dataset_size = dataset[0].shape[0]

  split = int(dataset_size * 0.9)
  indices = npr.permutation(dataset_size)
  training_idx, test_idx = indices[:split], indices[split:]
  training_dataset = [elem[:split] for elem in dataset]
  test_dataset = [elem[split:] for elem in dataset]

  training_dataset = dataset_cls(*training_dataset, device)
  training_loader = data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

  test_dataset = dataset_cls(*test_dataset, device)
  test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

  return training_dataset, training_loader, test_dataset, test_loader

def formatImageTensor(tensor):
  if tensor.dim() == 3:
    tensor = tensor.unsqueeze(-1)
  return tensor.permute(0,3,1,2).float()
