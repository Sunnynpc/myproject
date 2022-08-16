import torch,thop
from torch import nn

class Net_v1(nn.Module):
    def __init__(self):
        super(Net_v1, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(300*300*3,100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 96),
            nn.ReLU(),
            nn.Linear(96, 78),
            nn.ReLU(),
            nn.Linear(78, 64),
            nn.ReLU(),
            nn.Linear(64, 52),
            nn.ReLU(),
            nn.Linear(52,4)
        )

    def forward(self,x):
        return self.fc_layers(x)

class Net_v2(nn.Module):
    def __init__(self):
        super(Net_v2, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU()
        )
        self.out_layer1 = nn.Sequential(
            nn.Linear(256*4*4,4),
            nn.Sigmoid()
        )
        self.out_layer2 = nn.Sequential(
            nn.Linear(256*4*4,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        conv_out = self.conv_layers(x)
        conv_out = conv_out.reshape(-1,256*4*4)
        out1 = self.out_layer1(conv_out)
        out2 = self.out_layer2(conv_out)
        return out1,out2

class Res_Block(nn.Module):
    def __init__(self, c):
        super(Res_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c, c, 3,1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x) + x

class Pool(nn.Module):
    def __init__(self,c_in,c_out):
        super(Pool, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 2, padding=1,bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, padding=1,bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )
    def forward(self,x):
        return self.layer(x)

class Res_Net18(nn.Module):
    def __init__(self):
        super(Res_Net18, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            Res_Block(64),
            Res_Block(64),

            Pool(64, 128),
            Res_Block(128),

            Pool(128, 256),
            Res_Block(256),

            Pool(256, 512),
            Res_Block(512),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.out_layer1 = nn.Sequential(
            nn.Linear(512, 4, bias=True),
            nn.Sigmoid()
        )
        self.out_layer2 = nn.Sequential(
            nn.Linear(512, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.layer(x)
        conv_out = conv_out.reshape(-1, 512)
        out1 = self.out_layer1(conv_out)
        out2 = self.out_layer2(conv_out)
        return out1, out2
        # return conv_out

if __name__ == '__main__':
    # net = Net_v1()
    # x = torch.randn(3,300*300*3)
    # y = net(x)
    # print(y.shape)
    # flops, params = thop.profile(net, (x,))
    # print(flops)
    # print(params)
    net = Res_Net18()
    x = torch.randn(1,3,300,300)
    y = net(x)

    print(y[1])
    print(y[0])
