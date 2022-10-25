import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 感知机预测分类
# torch.normal() --返回正态分布的值.    (均值,标准差,size)
# torch.max(arr,1) 参数2为axis  --返回行最大的数
# torch.max()[0]， 只返回最大值的每个数   troch.max()[1]， 只返回最大值的每个索引


epoch = 200
n_data = torch.ones(100, 2)

x0 = torch.normal(2 * n_data, 1)
x1 = torch.normal(-2 * n_data, 1)
y0 = torch.ones(100)
y1 = torch.zeros(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1)).type(torch.LongTensor)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.out(x)
        return x


net = Net(2, 10, 2)
optim = torch.optim.SGD(net.parameters(), lr=0.005)
loss_func = torch.nn.CrossEntropyLoss()

for i in range(epoch):
    pred = net(x)

    loss = loss_func(pred, y)
    optim.zero_grad()
    loss.backward()

    optim.step()

    if i % 10 == 0:
        plt.cla()
        pred_val = torch.max(pred, 1)[1]  # pred.shape=(200,2)   pred_val类型为torch,有两个属性
        pred_y = pred_val.data.numpy()
        plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=pred_val, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == y.data.numpy()).astype(int).sum()) / float(y.data.numpy().size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()


torch.save(net,'classifier_net1.pth')
torch.save(net.state_dict(),'net1.parm.pth')
