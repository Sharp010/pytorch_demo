import numpy as np
import torch
import torchvision
from torch import nn
# 使用vae自变分编码器  解码编码mnist数据集
# vae 损失函数只能用均方误差（MSE）之类的粗略误差衡量
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
epoch=10
batch_size=64
LR=0.005
train_data=torchvision.datasets.MNIST("./mnist",train=True,
                                      transform=transforms.ToTensor(),
                                      download=False,
                                      )
train_data_loader=DataLoader(dataset=train_data,
                             batch_size=batch_size,
                             shuffle=True,)


class vae(nn.Module):
    def __init__(self):
        super(vae, self).__init__()
        self.encode=nn.Sequential(
            nn.Linear(28*28,128),
            nn.Tanh(),
            nn.Linear(128,32),
            nn.Tanh(),
            nn.Linear(32,3),
        )
        self.decode=nn.Sequential(
            nn.Linear(3,32),
            nn.Tanh(),
            nn.Linear(32,128),
            nn.Tanh(),
            nn.Linear(128,28*28),
            nn.Sigmoid()           # compress output to 0-1
        )
    def forward(self,input):
        encode=self.encode(input)
        decode=self.decode(encode)
        return encode,decode

net=vae()

# vae loss
loss_func=torch.nn.MSELoss()
optim=torch.optim.Adam(net.parameters(),lr=LR)

for i in range(epoch):
    total_loss=0
    for data,label in train_data_loader:
        # change shape
        b_x = data.view(-1, 28 * 28)  # batch x, shape (batch, 28*28)
        b_y = data.view(-1, 28 * 28)  # batch y, shape (batch, 28*28)

        encode, decode = net(b_x)

        loss=loss_func(decode,b_y)
        total_loss+=loss
        optim.zero_grad()
        loss.backward()
        optim.step()
    print("epoch {} | loss {:.4f}".format(i,total_loss))

torch.save(net,"vae_mnist_gen_model.pth")
# view
view_data=train_data.train_data[:20].view(-1,28*28).type(torch.FloatTensor)/255.
_,view=net(view_data)
for i in range(20):
    # plt.imshow(view.data[i].view(28,28).numpy(),cmap='gray')
    plt.imshow(np.reshape(view.data.numpy()[i], (28, 28)), cmap='gray')
    plt.show()
print("模型保存成功!")



























