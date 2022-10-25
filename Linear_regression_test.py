import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 单层感知机 进行线性拟合曲线
# nn,opt,loss,plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())


class perception(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_pred):
        super(perception, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.pred = torch.nn.Linear(n_hidden, n_pred)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.pred(x)
        return x


# loss and optim
perception = perception(n_feature=1, n_hidden=10, n_pred=1)
optim = torch.optim.SGD(perception.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

plt.ion()

epoch = 200
for i in range(epoch):
    pred = perception(x)
    # pred=perception.forward(x)
    loss = loss_func(pred, y)

    optim.zero_grad()
    # 若一次迭代中需要使用多次backward来调参,则使用loss.backward(retain_graph=True)
    # loss.backward(retain_graph=True)
    loss.backward()
    optim.step()

    if i % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), pred.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'LOSS=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()