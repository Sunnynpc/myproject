import torch
from torch import nn

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

        self.out_layer = nn.Sequential(
            nn.Linear(512, 1000, bias=True),
        )

    def forward(self, x):
        conv_out = self.layer(x)
        conv_out = conv_out.reshape(-1, 512)
        out = self.out_layer(conv_out)
        return out

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    net = Res_Net18()
    y = net(x)
    print(y.shape)