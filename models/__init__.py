from .Resnet50 import ResnetModel50
from .Resnet101 import ResnetModel101
from .VGG16 import VGG16
from .VGG19 import VGG19
from .Xception import Xception
from .MobileNet import MobileNet
from .MobileNetV2 import MobileNetV2
from .InceptionV3 import InceptionV3
from .ResNet50V2 import ResNet50V2
from .EfficientNetB7 import EfficientNetB7
from .DenseNet201 import DenseNet201

def get_model(name):
  if name == "resnet50":
    return ResnetModel50()
  elif name == "resnet101": # Don't work
    return ResnetModel101()
  elif name == "vgg16":
    return VGG16()
  elif name == "vgg19":
    return VGG19()
  elif name == "xception": # Wrong image size
    return Xception()
  elif name == "mobileNet":
    return MobileNet()
  elif name == "mobileNetV2": # Don't work
    return MobileNetV2()
  elif name == "inceptionV3": # Wrong image size
    return InceptionV3()
  elif name == "resNet50V2": # Don't work
    return ResNet50V2()
  elif name == "efficientNetB7": # Don't work
    return EfficientNetB7()
  elif name == "denseNet201": # Don't work
    return DenseNet201()
  else:
    raise ValueError("No model with name:", name)