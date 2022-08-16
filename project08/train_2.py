from dataset import Mydataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from net import Net
from net_2 import Mynet
from net import PetNet,FocalLoss
import os
import numpy as np

threshold = 0.5
mode_path = r'./param/net2.pt'
save_path = r'./param/net1.pt'
train_dataset = Mydataset('F:\data', is_train=True)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_dataset = Mydataset('F:\data',is_train=False)
test_loader = DataLoader(test_dataset,batch_size=5,shuffle=True)
# net1 = Net().cuda()
net = PetNet().cuda()

# net1.load_state_dict(torch.load(mode_path))
# print('Classification network loaded successfully')
# loss_fun = nn.NLLLoss()
loss_fun = FocalLoss()
opt = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.95, 0.85))

if os.path.exists(mode_path):
    net.load_state_dict(torch.load(mode_path))
    print('load success')

for epoch in range(1000000):
    sum_loss = 0.
    train_sum_sorce = 0.
    for i, (x, y) in enumerate(train_loader):
        net.train()
        # net2.train()
        x = x.cuda()
        y = y.long()
        y = y.cuda()
        # mask = net2(x)
        # a = torch.ones(26,1).cuda()
        # b = torch.zeros(26,1).cuda()
        # mask = torch.where(mask>0.5,a,b)
        # input = mask * x
        # out = net1(input)
        out,_ = net(x)
        out = out[:,0]

        loss = loss_fun(out, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sum_loss = sum_loss + loss.cpu().detach().item()
        pre = torch.argmax(out, dim=1)
        sorce = torch.mean(torch.eq(pre, y).float())
        train_sum_sorce += sorce
        train_sum_sorce =train_sum_sorce.cpu().detach().item()
        # print(out)
    avg_loss = sum_loss / len(train_loader)
    train_avg_score = train_sum_sorce / len(train_loader)
    print('avg_loss:', avg_loss, 'avg_score:', train_avg_score)
    # sum_sorce = 0.
    # test_sum_loss = 0.
    # for i, (data, labels) in enumerate(test_loader):
    #     net.eval()
    #     data, labels = data.cuda(), labels.cuda()
    #     labels = labels.long()
    #     test_out, mask = net(data)
    #     # mask = mask.squeeze()
    #     test_loss = loss_fun(test_out, labels)
    #
    #     test_sum_loss += test_loss.item()
    #     pre = torch.argmax(test_out, dim=1)
    #     sorce = torch.mean(torch.eq(pre, labels).float())
    #     sum_sorce += sorce
    #     sum_sorce = sum_sorce.cpu().detach().item()
    # test_avg_loss = test_sum_loss / len(test_loader)
    # test_avg_score = sum_sorce / len(test_loader)
    # print("test_avg_loss:", test_avg_loss, "test_avg_score:", test_avg_score)
    torch.save(net.state_dict(), mode_path)
    torch.save(opt.state_dict(),save_path)
    print('save success')
