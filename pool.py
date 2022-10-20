import torch
from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


class poll(nn.Module):
    def __init__(self):
        super(poll, self).__init__()
        self.poll = MaxPool2d(3, ceil_mode=True)

    def forward(self, input):
        return self.poll(input)


train_data = torchvision.datasets.CIFAR10("./data", False, transforms.ToTensor(), download=True)
writer = SummaryWriter("logs")
poll = poll()
print(input)
dataloader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
i = 0
for data in dataloader:
    inputs, targets = data
    writer.add_images("input", inputs, i)
    outputs = poll(inputs)
    writer.add_images("output", outputs, i)
    i += 1
    if i == 10:
        break
writer.close()
# input=torch.tensor([[1,2],[3,2]])
# output=torch.reshape(input,(-1,1,2,2))
# print(input)
# print(output)
# print(input.shape)
# print(output.shape)
