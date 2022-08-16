import time

import torch
from torch import nn

import cfg
from module import Net, ArcLoss
from torch import optim
from data import TestDataset
from torch.utils.data import DataLoader


def sim(x, y):
    pass


class Tester:

    def __init__(self):
        self._net = Net()
        self._net.eval()

    def __call__(self, *args, **kwargs):

        for _epoch in range(1000000000000):
            #
            self._net.load_state_dict(torch.load("w.pt"))

            self._dataset = TestDataset(self._net)
            self._test_dataloader = DataLoader(
                self._dataset,
                batch_size=cfg.test_batch_size,
            )

            #
            for _i, (_data, _target_featrue) in enumerate(self._test_dataloader):
                _feature = self._net(_data).cpu().detech()

                _rst = sim(_feature[:, None], self._dataset._feature_lib[None])

                _max_value = torch.max(_rst, dim=-1)
                _max_index = torch.argmax(_rst, dim=-1)

                _r = torch.sum(_target_featrue[_max_value > 0.5] == _max_index[_max_value > 0.5]) / len(
                    _target_featrue[_max_value > 0.5])

            time.sleep(60)
