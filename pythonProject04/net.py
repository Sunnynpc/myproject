import torch
from torch import nn

config = [
    [-1, 32, 1, 2],
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1]
]


class Block(nn.Module):
    def __init__(self, p_c, i, t, c, n, s):
        super(Block, self).__init__()
        # 每个重复的最后一次负责下采样和通道处理,i=n-1时进行操作
        self.i = i
        self.n = n

        _s = s if i == n-1 else 1
        _c = c if i == n-1 else p_c

        _p_c = p_c * t
        self.layer = nn.Sequential(
            nn.Conv2d(p_c, _p_c, 1, 1, bias=False),
            nn.BatchNorm2d(_p_c),
            nn.ReLU6(),
            nn.Conv2d(_p_c, _p_c, 3, _s, bias=False, padding=1, groups=_p_c),
            nn.BatchNorm2d(_p_c),
            nn.ReLU6(),
            nn.Conv2d(_p_c, _c, 1, 1, bias=False),
            nn.BatchNorm2d(_c)
        )
    def forward(self, x):
        if self.i == self.n - 1 :
            return self.layer(x)
        else:
            return self.layer(x) + x

class MobileNet_v2(nn.Module):
    def __init__(self, config):
        super(MobileNet_v2, self).__init__()
        self.config = config
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        self.Block = []
        p_c = config[0][1]
        for t, c, n, s in self.config[1:]:
            for i in range(n):
                self.Block.append(Block(p_c, i, t, c, n, s))
            p_c = c

        self.hidden_layer = nn.Sequential(*self.Block)

        self.output_layer = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(),
            nn.AvgPool2d(7, 1),
            nn.Conv2d(1280, 10, 1, 1)
        )

    def forward(self,x):
        out = self.input_layer(x)
        out = self.hidden_layer(out)
        out = self.output_layer(out)
        return out

if __name__ == '__main__':
    net = MobileNet_v2(config)
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.shape)
    print(net)

