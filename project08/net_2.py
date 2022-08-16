import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(26, 32, 2, 1, 1,bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AvgPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, 1, 1,bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1, 1,bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(224, 224, 3, 1),
            nn.ReLU(),
            nn.BatchNorm1d(224),
            nn.AvgPool1d(4),
        )
        self.fc = nn.Sequential(
            nn.Linear(224*78, 2),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = torch.cat((out1,out2,out3),dim=1)
        out = self.layer4(out4)
        out = out.reshape(-1,224*78)
        out = self.fc(out)
        return out


class Res_Block(nn.Module):
    def __init__(self, c):
        super(Res_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(c, c, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(c),
            nn.Conv1d(c, c, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(c),
        )

    def forward(self, x):
        return torch.cat((x, self.layer(x)), 1)


class DownSampling(nn.Module):
    def __init__(self, c):
        super(DownSampling, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(c, c, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(c),
        )

    def forward(self, x):
        return self.layer(x)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(26, 32, 3, 1,1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            Res_Block(32),
            DownSampling(64),
            # Res_Block(64),
            # DownSampling(128),
            # Res_Block(128),
            # DownSampling(256),
            # Res_Block(256),
        )
        self.out_layer = nn.Sequential(
            nn.Conv1d(64, 1, 3, 1,1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer(x)
        out = self.out_layer(out)
        # out = out.permute(0,2,1)
        return out

#
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.net1 = Net()
        self.net2 = Net2()

    def forward(self, x):
        x = x.permute(0,2,1)
        mask = self.net2(x)
        # a = torch.ones(1,632).cuda()
        # b = torch.zeros(1,632).cuda()
        # mask = torch.where(out > 0.5, a, b)
        out = mask * x
        out = self.net1(out)
        return out,mask


if __name__ == '__main__':
    net =Mynet()
    x = torch.randn(1, 632,26)
    out = net(x)
    print(out.shape)
