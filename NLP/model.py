import torch
from torch import nn
from torch.optim import Adam


class CBOW(nn.Module):

    def __init__(self):
        super().__init__()

        self._emb = nn.Parameter(torch.randn(240, 3))

        self._layer = nn.Linear(12, 3)

    def forward(self, x, x1):
        _e = self._emb[x]
        _y1 = self._emb[x1]
        _e = _e.reshape(-1, 12)
        _y = self._layer(_e)
        return _y,_y1


if __name__ == '__main__':
    net = CBOW()
    x = torch.randn(7,4)
    y = net(x.long())
    print(y.shape)
    # Adam(net.parameters())
