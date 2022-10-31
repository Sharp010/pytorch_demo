import  torch
from torch import  nn
from torchvision import  transforms
# lstm 判断句子中单词词性
#定义训练数据
training_data = [
("The cat ate the fish".split(), ["DET", "NN", "V", "DET", "NN"]),
("They read that book".split(), ["NN", "V", "DET", "NN"])
]
#定义测试数据
testing_data=[("They ate the fish".split())]

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
embedding_dim=10
hidden_dim=20       #
output_size=3

LR=0.02
epoch=1000
word_to_idx={}
label_to_idx={}
for data in training_data:
    for d in data[0]:
        if not d in word_to_idx.keys():
            word_to_idx[d]=len(word_to_idx)
    for l in data[1]:
        if not l in label_to_idx.keys():
            label_to_idx[l]=len(label_to_idx)
num_embedding=len(word_to_idx)  #9

def translate(list,to_idx):
    return torch.LongTensor([to_idx[i] for i in list])


class net(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,num_embedding,output_size):
        super(net, self).__init__()
        self.hidden_dim=hidden_dim
        # 获取词标号对应的词向量 一个词对应一个embedding_dim的向量
        self.embed=nn.Embedding(num_embedding,embedding_dim)
        self.lstm=nn.LSTM(embedding_dim,hidden_dim)
        self.linear=nn.Linear(hidden_dim,output_size)
        self.softmax=nn.Softmax()
    # def init_hidden(self):
    #     self.hidden=(torch.zeros(1,1,self.hidden_dim),
    #                  torch.zeros(1,1,self.hidden_dim))
    def forward(self,sentence):
        input=self.embed(sentence)  # shape=[x,10]
        # [x,1,10]->[x,1,hidden]->[x,1,output]->[x,output]   batch=1
        output,_=self.lstm(input.view(len(sentence),1,-1))
        return torch.nn.functional.log_softmax(self.linear(output).view(len(sentence),-1),dim=1)

net=net(embedding_dim=embedding_dim,hidden_dim=hidden_dim,
        num_embedding=num_embedding,output_size=output_size)
# net.init_hidden()
loss_func=nn.CrossEntropyLoss()
optim=torch.optim.SGD(net.parameters(),lr=LR,momentum=0.6)
#
# input,label=translate(training_data[0][0],word_to_idx).to(device),translate(training_data[0][1],label_to_idx).to(device)
# output=net(input)
# print(output.shape)


for i in range(epoch):
    net.train()
    for j in range(len(training_data)):
        # data_shape [x]
        data,labels=translate(training_data[j][0],word_to_idx).to(),translate(training_data[j][1],label_to_idx).to()
        outputs=net(data)

        loss=loss_func(outputs,labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

outputs=net(translate(testing_data[0],word_to_idx))
print(testing_data[0])
print(label_to_idx)
print(outputs.argmax(1))










