import torchvision.datasets
from torch import nn
import torch
import torch.nn
from torch.functional import F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, (5, 5), padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 5), padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, input):
        output = self.seq(input)
        return output
# input=torch.tensor([[1,2,3,4],
#                     [5,6,7,8],
#                     [1,2,3,5],
#                     [1,3,2,1]])
# kernel=torch.tensor([[1,2],
#                      [0,2]])
# input=torch.reshape(input,(1,1,4,4))
# kernel=torch.reshape(kernel,(1,1,2,2))
# output=F.conv2d(input,kernel,stride=1)
# print(output)
# train_data=torchvision.datasets.CIFAR10("./data",False,transforms.ToTensor(),download=True)
# writer=SummaryWriter("logs")
# model=Model()
# dataloader=DataLoader(dataset=train_data,batch_size=16,shuffle=True)
# i=1
# for data in dataloader:
#     imgs,targets=data
#     output=model(imgs)
#     writer.add_images("input1",imgs,i)
#     writer.add_images("output1",output,i)
#     i+=1
#     if i==5:
#         break
# writer.close()
