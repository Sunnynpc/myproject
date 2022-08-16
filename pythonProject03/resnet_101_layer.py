import torch
from torch import nn

class Res_Block(nn.Module):
    def __init__(self, c_in, c_out):
        super(Res_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1,1, padding=1, bias=False),
            nn.BatchNorm2d(c_in),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_in, 1, 1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x) + x



class Pool(nn.Module):
    def __init__(self,c_1,c_2, c_3, s):
        super(Pool, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_1, c_2, 3, stride=s, padding=1,bias=False),
            nn.BatchNorm2d(c_2),
            nn.ReLU(),
            nn.Conv2d(c_2, c_2, 3, 1, padding=1,bias=False),
            nn.BatchNorm2d(c_2),
            nn.ReLU(),
            nn.Conv2d(c_2, c_3, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(c_3),
            nn.ReLU()
        )
    def forward(self,x):
        return self.layer(x)

class Res_Net101(nn.Module):
    def __init__(self):
        super(Res_Net101, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            Pool(64, 64, 256, 1),
            Res_Block(256, 64),
            Res_Block(256, 64),

            Pool(256, 128, 512, 2),
            Res_Block(512, 128),
            Res_Block(512, 128),
            Res_Block(512, 128),

            Pool(512, 256, 1024, 2),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),
            Res_Block(1024, 256),

            Pool(1024, 512, 2048, 2),
            Res_Block(2048, 512),
            Res_Block(2048, 512),


            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.out_layer = nn.Sequential(
            nn.Linear(2048, 1000, bias=True),
        )

    def forward(self, x):
        conv_out = self.layer(x)
        conv_out = conv_out.reshape(-1, 2048)
        out = self.out_layer(conv_out)
        return out

if __name__ == '__main__':
    net = Res_Net101()
    print(net)