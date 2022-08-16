import torch
from torch.utils.data import DataLoader
from model import CBOW
from dataset import MyData
from torch import nn
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
os.environ['KMP_DUPLICATE_LIB_OK']='True'
save_path = r'./param/net1.pt'
train_dataset = MyData()
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
curr_words = train_dataset.curr_words

net = CBOW()
# loss_fun = nn.NLLLoss()
loss_fun = nn.MSELoss()
opt = torch.optim.Adam(net.parameters())
# summerWriter = SummaryWriter('logs')
#
if os.path.exists(save_path):
    net.load_state_dict(torch.load(save_path))
    print('load success')

for epoch in range(1000):
    sum_loss = 0.
    train_sum_sorce = 0.
    for i, (x, y) in enumerate(train_loader):
        net.train()
        x, y = x.cuda(), y.cuda()
        out,label = net(x.long(),y.long())
        loss = loss_fun(out, label)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sum_loss += loss.item()

        # print(out)
    avg_loss = sum_loss / len(train_loader)
    print('avg_loss:', avg_loss)
    torch.save(net.state_dict(),save_path)
    print('save success')

# plt.rcParams['font.sans-serif']=['SimHei'] # 显示中文标签
# plt.rcParams['axes.unicode_minus']=False
# emb = net._emb.detach().numpy()
# x,y,z = emb[:,0], emb[:,1],emb[:,2]
# Axes3D = plt.gca(projection='3d')
# Axes3D.scatter(x, y, z)
# for i in range(len(x)):
#     Axes3D.text(x[i],y[i],z[i],curr_words[i])
# plt.show()
