import numpy as np
import torch
from torch.utils.data import Dataset
import jieba


class MyData(Dataset):
    def __init__(self):
        path_txt = './test.txt'
        txt = open(path_txt, 'r', encoding='utf-8').read()
        excludes = ["，", "：", "“", "。", "”", "、", "；", " ", "\n", "…", "‘", "’"]
        txt = txt.replace('\n', '')
        sentence = txt.split('。')
        words = jieba.lcut(txt)
        # 词表
        self.curr_words = []
        for word in words:
            if word in excludes or word.isdigit():
                continue
            else:
                self.curr_words.append(word)
        # 词表去重
        self.curr_words = list(set(self.curr_words))
        self.curr_words.sort()
        # 句子
        self.sentences = []
        for s in sentence:
            s = jieba.lcut(s)
            word = []
            for w in s:
                if w in excludes or w.isdigit():
                    continue
                else:
                    word.append(w)
            self.sentences.append(word)
        self.sentences = self.sentences[:-1]
        self.dataset =[]
        for sentence in self.sentences:
            word_index = []
            for word in sentence:
                for i, w in enumerate(self.curr_words):
                    if w == word:
                        word_index.append(i)
                    else:
                        continue
            word_index = np.array(word_index,dtype='float32')
            if len(sentence) <65:
                word_index = np.pad(word_index,(2,63-len(sentence)))
            for i in range(2, len(sentence)+2):
                self.dataset.append((np.concatenate((word_index[i - 2:i], word_index[i + 1:i + 3]), axis=0), word_index[i]))
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        data = self.dataset[index]
        return torch.tensor(data[0]),torch.tensor(data[1]),self.curr_words

if __name__ == '__main__':
    data = MyData()
    print(data[0][1].shape)