import torch
from torch import nn
from torch.functional import F

import cfg


class Net(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # TODO

        feature = torch.randn(cfg.train_batch_size, cfg.feature_dim)
        return feature


class ArcLoss(nn.Module):

    def __init__(self, feature_dim, cls_dim, m):
        super().__init__()
        self._m = m

        self._w = nn.Parameter(torch.randn(feature_dim, cls_dim))
        self._loss_fn = nn.CrossEntropyLoss()

    def forward(self, feature):
        _out = feature @ self._w

        _feature_normal_sum = torch.norm(feature, dim=-1)[:,None]
        _feature_normal = F.normalize(feature)
        # _feature_normal = feature / _feature_normal_sum
        #
        _w_normal = F.normalize(self._w)
        _angle = torch.acos(torch.matmul(_feature_normal, _w_normal))
        #
        a = _feature_normal_sum * torch.cos(_angle + self._m)
        _up_i = torch.exp(_feature_normal_sum * torch.cos(_angle + self._m))
        _down_i = torch.exp(_feature_normal_sum * torch.cos(_angle))
        _down_sum = torch.sum(_down_i, dim=-1, keepdim=True)
        #
        _v = _up_i / (_down_sum - _down_i + _up_i)

        return _v

if __name__ == '__main__':
    arc = ArcLoss(128,3488,0.01)
    data = torch.randn(21,128)
    out = arc(data)
    print(data)
    print(out.shape)