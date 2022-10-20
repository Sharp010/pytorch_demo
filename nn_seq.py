import torchvision.datasets
from torch import nn
import torch
import torch.nn
from torch.functional import F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10("./data", True, transform=transforms.ToTensor(), download=True)
DataLoader = DataLoader(dataset=train_data, batch_size=1)


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


# model=Model()
# input=torch.ones((1,3,32,32))
# print(model(input).shape)
loss = nn.CrossEntropyLoss()
model = Model()
optim = torch.optim.SGD(model.parameters(), lr=0.005)
for k in range(20):
    i = 0
    sum = 0
    for data in DataLoader:
        img, target = data
        loss_val = loss(model(img), target)
        sum += loss_val
        optim.zero_grad()
        loss_val.backward()
        optim.step()
        i += 1
        if i == 200:
            print("loss=", sum)
            break
