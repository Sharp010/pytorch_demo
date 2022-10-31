import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda")
epoch = 15
batch_size = 10


# 独热编码
# 输入x代表默认的torchvision返回的类比值，class_count类别值为10
def one_hot(x, class_count=10):
    return torch.eye(class_count)[x, :]  # 切片选取，第一维选取第x个，第二维全要


train_data = torchvision.datasets.MNIST("./mnist", train=True,
                                        transform=transforms.ToTensor(), download=True, target_transform=one_hot)

train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        # input
        # 10个100维向量
        # 10个label->emb->10个10维向量
        super(Generator, self).__init__()
        # self.label_emb=nn.Embedding(10,10)
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, label):
        # [10,100]
        z = z.view(z.size(0), 100)
        # c=self.label_emb(label)
        # TODO []
        # [10,110]
        x = torch.cat([z, label], 1)
        out = self.model(x)
        return out.view(-1, 28, 28)  # [10,28,28] 0-1


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(28 * 28 + 10, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        # [10,784]
        x = x.view(x.size(0), 28 * 28)
        # c=self.label_emb(label)
        # [10,794]
        x = torch.cat([x, label], 1)
        out = self.model(x)
        return out.squeeze()  # [10]  0-1


d = Discriminator().to(device)
g = Generator().to(device)
loss = torch.nn.BCELoss().to(device)
d_optim = torch.optim.Adam(d.parameters(), lr=0.0005)
g_optim = torch.optim.Adam(g.parameters(), lr=0.0005)

for i in range(epoch):
    d_loss_total1 = 0
    d_loss_total2 = 0
    g_loss_total = 0
    # imgs[10,28,28] labels[10,10]
    for imgs, labels in train_data_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        imgs = imgs.reshape(batch_size, -1)  # (10,784)

        real_labels = torch.ones(batch_size).to(device)
        fake_labels = torch.zeros(batch_size).to(device)

        # 判别器
        # real
        d_optim.zero_grad()
        real_validity = d(imgs, labels)
        d_loss_real = loss(real_validity, real_labels)
        d_loss_real.backward()
        # fake
        z = torch.randn(batch_size, 100).to(device)

        fake_imgs = g(z, labels)
        fake_validity = d(fake_imgs.detach(), labels)
        d_loss_fake = loss(fake_validity, fake_labels)
        d_loss_fake.backward()

        d_loss_total1 += d_loss_fake
        d_loss_total2 += d_loss_real
        # bp
        d_optim.step()
        # 生成器
        g_optim.zero_grad()
        gen_validity = d(fake_imgs, labels)
        g_loss = loss(gen_validity, real_labels)
        g_loss_total += g_loss
        # bp
        g_loss.backward()
        g_optim.step()
    print(
        "epoch {} | d_loss1_fake {:.4f} | d_loss2_real {:.4f} |  g_loss  {:.4f}".format(i, d_loss_total1, d_loss_total2,
                                                                                        g_loss_total))
torch.save(g, "G_1.pth")
torch.save(d, "D_1.pth")
print("模型保存成功!")


#test
device='cuda' if torch.cuda.is_available() else 'cpu'
g=torch.load("./G_1.pth",map_location=torch.device('cpu'))
z = torch.randn(1, 100).to(device)
img = g(z, one_hot([5]).to(device))
img = 0.5 * img + 0.5
transforms.ToPILImage()(img)