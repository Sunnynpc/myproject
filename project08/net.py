import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self,in_chanel):
        super(ResBlock, self).__init__()
        m_chanel = in_chanel*2
        self.layer = nn.Sequential(
            nn.Conv1d(in_chanel,m_chanel,1,1,0),
            nn.ReLU(),
            nn.BatchNorm1d(m_chanel),
            nn.Conv1d(m_chanel,m_chanel,3,1,1,groups=m_chanel),
            nn.ReLU(),
            nn.BatchNorm1d(m_chanel),
            nn.Conv1d(m_chanel,in_chanel,1,1,0)
        )
    def forward(self,x):
        return self.layer(x)+x
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(26,16,1,1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            ResBlock(16),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 1, 1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            ResBlock(32),
            nn.MaxPool1d(2),
            ResBlock(32),
            nn.MaxPool1d(2),
            ResBlock(32),
            nn.MaxPool1d(2),
            ResBlock(32),
            nn.MaxPool1d(2),
            ResBlock(32),
            nn.MaxPool1d(2),
            nn.Conv1d(32,1,9),
            nn.Flatten(),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.layer(x)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(26,16, 1, 1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            ResBlock(16),
            ResBlock(16),
            ResBlock(16),
            nn.Conv1d(16, 1, 1),
            nn.Softmax(2)
        )
    def forward(self,x):
        return self.layer(x)
class PetNet(nn.Module):

    def __init__(self):
        super().__init__()

        self._cls_net = Net()
        self._mask_net = Net2()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        _mask = self._mask_net(x)
        _y = x * _mask

        _y = self._cls_net(_y)
        return _y, _mask

class FocalLoss(nn.Module):

    def __init__(self, n=1):
        super().__init__()
        self._n = n

    def forward(self, y, target):

        _p_y = y[target == 1]
        _p_target = target[target == 1]

        _n_y = y[target == 0]
        _n_target = target[target == 0]

        _p_loss = -_p_target * torch.log(_p_y) * (1 - _p_y) ** self._n
        _n_loss = -(1 - _n_target) * torch.log(1 - _n_y) * _n_y ** self._n


        _loss = torch.mean(torch.cat([_p_loss, _n_loss], dim=-1))

        return _loss


if __name__ == '__main__':
    net = PetNet()
    x = torch.randn(20,632,26)
    out = net(x)
    print(out[1].shape)