from datasets.mnist import load_mnist
from .imagenet import load_imagenet
from .cifar10 import load_cifar10
from .cifar100 import load_cifar100
from .mnist import load_mnist

def get_dataset(name, num_images=1, labels=None):
  if name == "imagenet":
    return load_imagenet(num_images, labels)
  elif name == "cifar10":
    return load_cifar10(num_images)
  elif name == "cifar100":
    return load_cifar100(num_images)
  elif name == "mnist":
    return load_mnist(num_images)
  else:
    raise ValueError("No dataset with name:", name)