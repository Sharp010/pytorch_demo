import numpy as np
import cv2
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# 用[]包住
transforms_compose = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

writer = SummaryWriter("logs")

# imga=Image.open("data/train/ants/6240329_72c01e663e.jpg")
# train_data[idx]获取到img,target(为对应类别在train_data.classes的idx)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transforms_compose, download=True)
dataloader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0)
i = 0
for data in dataloader:
    imgs, targets = data
    i += 1
    if i == 11:
        break
    writer.add_images("test_data", imgs, i)

# writer.add_image()  和 writer.add_images()
# writer.add_image("test",transforms.ToTensor()(img),2)
writer.close()
