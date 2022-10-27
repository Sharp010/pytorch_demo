import time

import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import  transforms
import cv2 as cv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# 缺点 :  mnist 数据集均为纯黑底白字  对于不是纯黑底白字的数字识别率极低

# conv2d relu maxpool conv2d relu maxpool linear
# adam crossEntropyLoss
# nn.Flatten() 默认从dim=1出展开 (60,7,7) 展开为(60,49)
# nn.Flatten() 可以用img.view(img.size(),-1) 代替,把seq拆成两个

epoch = 3

train_data = torchvision.datasets.MNIST("./mnist", True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST("./mnist", False, transform=torchvision.transforms.ToTensor(), download=True)


# # image shape =(1,28,28)
#
train_dataLoader=DataLoader(dataset=train_data,batch_size=60,num_workers=0)
test_dataLoader=DataLoader(dataset=test_data,batch_size=60,num_workers=0)
#
convert = torchvision.transforms.Compose([transforms.Grayscale(),transforms.Resize((28, 28)),
                                          transforms.ToTensor()])
img=convert(Image.open("./num_img_data/9/num_9.png"))
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


nn=torch.load("Handwritten_numerals-recognition_model.pth",map_location=torch.device('cpu'))
loss_func=torch.nn.CrossEntropyLoss()
optim=torch.optim.Adam(nn.parameters(),lr=0.02)

start_time = time.time()
for i in range(epoch):
    nn.train()
    for imgs,targets in train_dataLoader:
        preds=nn(imgs)
        loss=loss_func(preds,targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
    nn.eval()
    print(nn(img.view(1,1,28,28)))
    with torch.no_grad():
        total_correct=0
        for imgs,targets in test_dataLoader:
            preds=nn(imgs)
            loss=loss_func(preds,targets)
            # preds-shape=[60,10]
            # targets-shape=[60]
            total_correct+=(preds.argmax(1)==targets).sum()
        print("第{}次训练-测试:accuracy={}".format(i,total_correct/len(test_data)))
torch.save(nn,"Handwritten_numerals-recognition_model.pth")
print("模型保存成功!")


# test model
# Handwritten_numerals-recognition_model.pth is trained on colab
# img must be transformed to Tensor
classifier = torch.load("Handwritten_numerals-recognition_model.pth", map_location=torch.device('cpu'))
img = Image.open("num_img_data/1/num_1.png")
img = img.convert("L")
convert = torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28)),
                                          torchvision.transforms.ToTensor()])
img = convert(img)
img = torch.reshape(img, (1, 1, 28, 28))
target = classifier(img)
print(target)
print("target-num={}".format(target.argmax()))
