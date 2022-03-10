from .Resnet50 import ResnetModel50
from .Resnet101 import ResnetModel101

def get_model(name):
  if name == "resnet50":
    return ResnetModel50
  elif name == "resnet101":
    return ResnetModel101
  else:
    raise ValueError("No model with name:", name)