import numpy as np
from torch.utils.data import Dataset

import cfg


class MyDataset(Dataset):

    def __init__(self):
        self._dataset = []

    def __len__(self):
        # TODO
        return cfg.train_batch_size * 2

    def __getitem__(self, item):
        return np.random.randn(cfg.train_seq_len, cfg.input_dim), \
               np.ones(cfg.cls_dim)


class TestDataset(Dataset):

    def __init__(self, net):
        self._net = net

        # 建立特征库
        # 1.读取测试所有的标签数据（去重）（词向量）
        # 2.送入到Net得到特征,放入到_feature_map
        self._feature_map = {
            "脂肪乳": np.random.randn(cfg.feature_dim),
            "羟乙基淀粉": np.random.randn(cfg.feature_dim)
        }

        self._feature_lib = np.stack(self._feature_map.values())

        # 建立测试数据对
        # 1.读取所有的数据对
        # 2.比较标签数据的名称,得到输入数据和标签特征

        # for _ in 所有数据对
        self.dataset = [
            [
                np.random.randn(cfg.train_seq_len, cfg.input_dim),  # 输入数据词向量
                0  # 该数据在 特征库里面的索引
            ]
        ]

    def __len__(self):
        # TODO
        return cfg.train_batch_size

    def __getitem__(self, item):
        data = self.dataset[item]
        return data
