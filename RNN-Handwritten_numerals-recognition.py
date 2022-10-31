import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torchvision
from torch import  nn
from torch.utils.data import  DataLoader
# rnn mnist

# 缺点 :  mnist 数据集均为纯黑底白字  对于不是纯黑底白字的数字识别率极低

lr=0.02
epoch=3
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# data
transform=transforms.Compose([transforms.ToTensor()])
train_data=torchvision.datasets.MNIST("./mnist",train=True,transform=transform)
test_data=torchvision.datasets.MNIST("./mnist",train=False,transform=transform)
# test
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels.numpy()[:2000]    # covert to numpy array
# data loader
train_dataloader=DataLoader(dataset=train_data,batch_size=64,shuffle=True)
test_dataloader=DataLoader(dataset=test_data,batch_size=64,shuffle=True)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn=nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out=nn.Linear(64,10)
    def forward(self,input):
        # output=(batch_size,step,10scores)
        output,_=self.rnn(input)
        # choose last step output
        return self.out(output[:,-1,:])

# model=torch.load("RNN-Handwritten_numerals-recognition_model.pth").to(device)
# loss_func=torch.nn.CrossEntropyLoss()
# optim=torch.optim.Adam(model.parameters(),lr=lr)
#
# for i in range(epoch):
#     total_loss=0
#     model.train()
#     for imgs,targets in train_dataloader:
#         imgs=imgs.to(device)
#         targets=targets.to(device)
#         # imgs.shape=[64,1,28,28]
#         # imgs.view(-1,28,28).shape=[64,28,28]
#
#         # preds.shape=[64,10]
#         preds=model(imgs.view(-1,28,28))
#         loss=loss_func(preds,targets)
#         total_loss+=loss
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#     # lr modify
#     optim.param_groups[0]['lr'] *= 0.3
#     model.eval()
#     with torch.no_grad():
#         total_correct=0
#         for imgs,targets in test_dataloader:
#             imgs=imgs.to(device)
#             targets=targets.to(device)
#             preds=model(imgs.view(-1,28,28)).argmax(1)
#             total_correct+=(preds==targets).sum()
#     print("round ",i," | loss %.5f"% total_loss," | accuracy %.5f" % (total_correct/len(test_data)))
#
# torch.save(model,"RNN-Handwritten_numerals-recognition_model.pth")
# print("模型保存成功!")

# # 检测
classifier = torch.load("RNN-Handwritten_numerals-recognition_model.pth", map_location=torch.device('cpu'))

convert = torchvision.transforms.Compose([transforms.Resize((28, 28)),
                                          transforms.ToTensor()])
# data_test=torchvision.datasets.ImageFolder("./num_img_data",
#                                              transform=convert,
#                                              loader=datasets.folder.default_loader)
# data_loader=DataLoader(dataset=data_test,batch_size=1,shuffle=True)
# # print(data_loader.shape)
# for img,_ in data_loader:
#     pred=classifier(img.view(1,28,28))
#     print(pred)
#     plt.imshow(img.view(28,28))
#     plt.title(pred.argmax())
#     plt.show()

img=Image.open("num_img_data/2/img_3.png").convert('L')
img=convert(img)

# 测试数据
plt.imshow(img.view(28,28),cmap='gray')
plt.title("img {}".format(classifier(img.view(1,28,28))))
plt.show()
print(img)
# mnist
plt.imshow(test_x[1],cmap='gray')
plt.title("img {}".format(classifier(test_x[1].view(1,28,28))))
plt.show()
print(test_x[1])





