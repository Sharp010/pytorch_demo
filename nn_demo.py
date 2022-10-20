import time

import torchvision.datasets
from torch import nn
import torch
import torch.nn
from torch.functional import F
from torchvision import transforms
from nn_conv import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# 加载cifar10数据集
train_data = torchvision.datasets.CIFAR10("./data", True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("./data", False, transform=transforms.ToTensor(), download=True)
# 数据集大小
test_data_size = len(test_data)
# dataloader数据加载器
train_dataLoader = DataLoader(dataset=train_data, batch_size=64, num_workers=0)
test_dataLoader = DataLoader(dataset=test_data, batch_size=64, num_workers=0)
# 交叉熵损失函数
loss = nn.CrossEntropyLoss()
model = torch.load("model_tarin_final.pth").cuda()
optim = torch.optim.SGD(model.parameters(), lr=0.01)
round = 10
start_time = time.time()
for i in range(round):
    # 训练
    print("第{}轮训练开始".format(i + 1))
    model.train()
    cnt = 0
    for data in train_dataLoader:
        imgs, targets = data
        outputs = model(imgs)
        loss_val = loss(outputs, targets)
        optim.zero_grad()
        loss_val.backward()
        # 对{卷积核}中的参数进行优化
        optim.step()
        cnt += 1
        if cnt % 100 == 0:
            end_time = time.time()
            print("训练时间:{}".format(end_time - start_time))
            print("第 {} 次训练loss: {}".format(cnt, loss_val))
    # 测试
    model.eval()
    with torch.no_grad():
        correct_cnt = 0
        for data in test_dataLoader:
            imgs, targets = data
            outputs = model(imgs)
            correct_cnt += (outputs.argmax(1) == targets).sum()
        print("test 正确率:{}".format(correct_cnt / test_data_size))
# 保存模型
torch.save(model, "model_tarin_final.pth")
print("模型保存成功！")
