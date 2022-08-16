import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class Mydataset(Dataset):
    def __init__(self, root_path, is_train=True):
        self.path = root_path
        self.file_dir = 'train' if is_train == True else 'test'
        self.dataset = []
        for label, filename in enumerate(os.listdir(os.path.join(root_path, self.file_dir))):
            file_path = os.path.join(self.path, self.file_dir, filename)
            file = open(file_path, 'r+')
            data = file.readlines()
            for line in data:
                self.dataset.append((line[:-1], label))
        # self.file_name = 'train_data.txt' if is_train == True else 'test_data.txt'
        # self.dataset = []
        # data_file = open(os.path.join(self.path,self.file_name),'r')
        # data = data_file.readlines()
        # for line in data:
        #     self.dataset.append((line[2:-1],line[0]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        data_item = []
        for i in data[0]:
            data_number = ord(i) - 65
            data_one_hot = np.zeros(26)
            data_one_hot[data_number] = 1
            data_item.append(data_one_hot)
        if len(data[0]) < 632:
            for j in range(632 - len(data[0])):
                data_one_hot2 = np.zeros(26)
                data_item.append(data_one_hot2)
        # y_one_hot = np.zeros(2)
        # y_one_hot[data[1]] = 1
        return np.float32(data_item), np.float32(data[1])


if __name__ == '__main__':
    dataset = Mydataset('F:\data')
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    for i, (x, y) in enumerate(dataloader):
        print(x.shape)
        print(y.shape)
