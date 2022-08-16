from net import Net
from dataset import Mydataset
import torch
from torch import nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import os

save_path = r'./param/net.pt'
train_dataset = Mydataset('F:\data', is_train=True)
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)

test_dataset = Mydataset('F:\data', is_train=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

net = Net().cuda()
# loss_fun = nn.NLLLoss()
loss_fun = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=0.01)
# summerWriter = SummaryWriter('logs')
#
if os.path.exists(save_path):
    net.load_state_dict(torch.load(save_path))
    print('load success')

for epoch in range(1000000):
    sum_loss = 0.
    train_sum_sorce = 0.
    for i, (x, y) in enumerate(train_loader):
        net.train()
        x, y = x.cuda(), y.cuda()
        x = x.permute(0, 2, 1)
        out = net(x)
        y = y.long()
        loss = loss_fun(out, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sum_loss += loss.item()
        pre = torch.argmax(out, dim=1)
        sorce = torch.mean(torch.eq(pre, y).float())
        train_sum_sorce += sorce
        # print(out)
    avg_loss = sum_loss / len(train_loader)
    train_avg_score = train_sum_sorce / len(train_loader)
    print('avg_loss:', avg_loss, 'avg_score:', train_avg_score.item())
    # sum_sorce = 0.
    # test_sum_loss = 0.
    # for i,(data,labels) in enumerate(test_loader):
    #     net.eval()
    #     data,labels = data.cuda(),labels.cuda()
    #     data = data.permute(0, 2, 1)
    #     labels = labels.long()
    #     test_out = net(data)
    #     test_loss = loss_fun(test_out,labels)
    #
    #     test_sum_loss += test_loss.item()
    #     pre = torch.argmax(test_out,dim=1)
    #     sorce = torch.mean(torch.eq(pre, labels).float())
    #     sum_sorce+= sorce
    # test_avg_loss = test_sum_loss/len(test_loader)
    # test_avg_score = sum_sorce/len(test_loader)
    # # summerWriter.add_scalars("loss",{"train_loss":avg_loss,"test_loss":test_avg_loss},epoch)
    # # summerWriter.add_scalar("test_avg_score", test_avg_score, epoch)
    # print("epoch:",epoch,"test_avg_loss:",test_avg_loss,"test_avg_score:", test_avg_score.item())
    torch.save(net.state_dict(), save_path)
    print('save success')
