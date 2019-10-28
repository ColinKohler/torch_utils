import gc
import torch

def memReport():
  for obj in gc.get_objects():
    if torch.is_tensor(obj):
      print(type(obj), obj.size(), obj.device)
