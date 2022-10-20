import torch

from nn_conv import *
from PIL import Image

data = torchvision.datasets.CIFAR10("./data", True, download=True)
convert = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                          torchvision.transforms.ToTensor()])
img = Image.open("./img_2.png")
# png四通道
img = img.convert("RGB")
img = convert(img)
# 变换格式
img = torch.reshape(img, (1, 3, 32, 32))
# 加载模型
model = torch.load("model_tarin_final.pth")
target = model(img)
print(target)
# print(target.shape)
# tensor张量
print(data.classes[target.argmax()])
