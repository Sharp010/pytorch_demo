import matplotlib.pyplot as plt
import numpy
import numpy as np
import cv2 as cv
import torch.utils.data
# torch
    # 图像处理 opencv+PIL+transforms
    # 数据统一加载  dataloader
    # 可视化 matplotlib
    # 记录 tensorboard
# torch.unsqueeze()维度扩充
# torch.squeeze()删除大小为1的维度
# for index,value in enumerate(x):

# transform.ToTensor() 将图像0-255 归一化 0-1
# transform.Normalize([mean1,mean2,mean3],[std1,std2,std3]) 将三个通道的数据标准化
# mean ,std 都为0.5 则将数据归一化为-1->1

# data.TensorDataSet(x,y) 生成数据

# torchvision.models 提供训练好的models

import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from PIL import Image
        # # tensorboard
        # # 用[]包住
        # transforms_compose = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        # writer = SummaryWriter("logs")
        # # imga=Image.open("data/train/ants/6240329_72c01e663e.jpg")
        # # train_data[idx]获取到img,target(为对应类别在train_data.classes的idx)
        # test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transforms_compose, download=True)
        # dataloader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0)
        # i = 0
        # for data in dataloader:
        #     imgs, targets = data
        #     i += 1
        #     if i == 11:
        #         break
        #     writer.add_images("test_data", imgs, i)
        # # writer.add_image()  和 writer.add_images()
        # # writer.add_image("test",transforms.ToTensor()(img),2)
        # writer.close()
# dataloader
# import torch
# arr=torch.from_numpy(np.asarray([[1,2,3,4],[5,6,4,3],[5,6,3,2],[2,3,4,5]]))
# transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

it=iter([1,2,3,6,2,23])
print(next(it))
print(next(it))
