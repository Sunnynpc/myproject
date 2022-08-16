import os
import random

positive_path = r'F:\data\train\1.txt'
out_positive_path = r'F:\data\train_s\1.txt'

positive_file = open(positive_path,'r+')
positive_data = positive_file.readlines()
# print(positive_data)
out_positive_file = open(out_positive_path,'w')
for line in positive_data:
    out_positive_file.write(line)
for i in range(10000):
    cut_datas = random.sample(positive_data, 2)
    cut_datas[0] = cut_datas[0][:-1]
    cut_datas[1] = cut_datas[1][:-1]
    len_key1 = random.randint(70, len(cut_datas[0])//2)
    len_key2 = random.randint(70, len(cut_datas[1])//2)

    index_key1 = random.randint(0, len(cut_datas[0]) - len_key1)
    index_key2 = random.randint(0, len(cut_datas[1]) - len_key2)

    cut_1 = cut_datas[0][index_key1:index_key1 + len_key1]
    cut_2 = cut_datas[1][index_key2:index_key2 + len_key2]
    new_data = cut_1 + cut_2 + '\n'

    out_positive_file.write(new_data)

out_positive_file.close()
















