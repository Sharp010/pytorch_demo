import torchvision
from torch import nn
vgg16=torchvision.models.vgg16()
print(vgg16)
vgg16.classifier[6]=nn.Conv2d()