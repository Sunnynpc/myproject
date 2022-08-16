import torch
from torch import nn

import cfg
from module import Net, ArcLoss
from torch import optim
from data import MyDataset
from torch.utils.data import DataLoader


class Trainer:

    def __init__(self):
        self._net = Net()
        self._loss_fn = ArcLoss()
        ps=[]
        ps.extend(self._net.parameters())
        ps.extend(self._loss_fn.parameters())

        self._opt = optim.Adam(ps)

        self._train_dataloader = DataLoader(
            MyDataset(),
            batch_size=cfg.train_batch_size,
            shuffle=True
        )



    def __call__(self):

        for _epoch in range(1000000000000):

            for _i, (_data, _target) in enumerate(self._train_dataloader):
                _feature = self._net(_data)
                _loss = self._loss_fn(_feature, _target)

                self._opt.zero_grad()
                _loss.backward()
                self._opt.step()
