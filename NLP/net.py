import torch
from torch import nn
from torch.optim import Adam


class CBOW(nn.Module):

    def __init__(self):
        super().__init__()

        self._emb = nn.Parameter(torch.randn(49, 3))
        self._layer = nn.Linear(12, 3)

    def forward(self, x, x1):
        _e = self._emb[x]#(N,4,3)
        _e = _e.reshape(-1, 12)#(N,3)
        _y = self._layer(_e)
        _y1 = self._emb[x1]#(N,3)
        return _y,_y1


if __name__ == '__main__':
    net = CBOW()
    a = torch.randint(10,(2,4))
    print(a)
    a1 = torch.randint(10,(2,))
    b,b1 = net(a,a1)
    print(b.shape)
    print(b1.shape)
    emb = net._emb
    print(emb)