import time

import torch
import torchvision
from PIL import Image
from torch import nn
import cv2 as cv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# conv2d relu maxpool conv2d relu maxpool linear
# adam crossEntropyLoss
# nn.Flatten() 默认从dim=1出展开 (60,7,7) 展开为(60,49)

epoch = 8

train_data = torchvision.datasets.MNIST("./mnist", True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST("./mnist", False, transform=torchvision.transforms.ToTensor(), download=True)


# # image shape =(1,28,28)
#
# train_dataLoader=DataLoader(dataset=train_data,batch_size=60,num_workers=0)
# test_dataLoader=DataLoader(dataset=test_data,batch_size=60,num_workers=0)
#
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, (5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, (5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(1568, 32),
            nn.Linear(32, 10)
        )

    def forward(self, img):
        return self.seq(img)


# nn=cnn()
# loss_func=torch.nn.CrossEntropyLoss()
# optim=torch.optim.Adam(nn.parameters(),lr=0.02)
#
# start_time = time.time()
# for i in range(epoch):
#     nn.train()
#     cnt=0
#     print("第{}轮训练开始".format(i + 1))
#     for imgs,targets in train_dataLoader:
#         preds=nn(imgs)
#         loss=loss_func(preds,targets)
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#
#         cnt += 1
#         if cnt % 1000 == 0:
#             end_time = time.time()
#             print("训练时间:{}".format(end_time - start_time))
#             print("第 {} 次训练loss: {}".format(cnt, loss))
#     nn.eval()
#     with torch.no_grad():
#         if i%2==0:
#             total_correct=0
#             for imgs,targets in test_dataLoader:
#                 preds=nn(imgs)
#                 loss=loss_func(preds,targets)
#                 # preds-shape=[60,10]
#                 # targets-shape=[60]
#                 total_correct+=(preds.argmax(1)==targets).sum()
#             print("第{}次训练-测试:accuracy={}".format(i,total_correct/len(test_data)))
# torch.save(nn,"Handwritten_numerals-recognition_model.pth")
# print("模型保存成功!")


# test model
# Handwritten_numerals-recognition_model.pth is trained on colab
# img must be transformed to Tensor
classifier = torch.load("Handwritten_numerals-recognition_model.pth", map_location=torch.device('cpu'))
img = Image.open("./num_8.png")
img = img.convert("L")
convert = torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28)),
                                          torchvision.transforms.ToTensor()])
img = convert(img)
img = torch.reshape(img, (1, 1, 28, 28))
target = classifier(img)
print("target-num={}".format(target.argmax()))
