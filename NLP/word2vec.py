import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import torch
from torch.nn import functional as F
import os

output_dir ='./param'
path_txt = './test.txt'
txt = open(path_txt, 'r', encoding='utf-8').read()
excludes = ["，", "：", "“", "。", "”", "、", "；", " ", "\n", "…", "‘", "’", "（", "）"]
txt= txt.replace('\n','')
sentence = txt.split('。')
words = jieba.lcut(txt)
sentences = []
for s in sentence:
    s=jieba.lcut(s)
    word=[]
    for w in s :
        if w in excludes or w.isdigit():
            continue
        else:
            word.append(w)

    sentences.append(word)
res = max(sentences, key=len, default='')
print(sentences)
print(len(res))


curr_words = []
for word in words:
    if word in excludes or word.isdigit():
        continue
    else:
        curr_words.append(word)
curr_words1= list(set(curr_words))
curr_words1.sort()
dict={word:index for index,word in enumerate(curr_words1)}
# print(curr_words)
print(len(curr_words1))
# word_index =[]
# for word in curr_words:
#     for i,w in enumerate(curr_words1):
#         if w==word:
#             word_index.append(i)
#         else:
#             continue
# word_index = torch.tensor(word_index)
# one_hot_word=F.one_hot(word_index)
# print(one_hot_word)
# sentences = LineSentence(sentences)
# model = Word2Vec(sentences=sentences, vector_size=259, epochs=10, min_count=1)
#
# # 保存模型
# model_file = os.path.join(output_dir, 'model.w2v')
# model.save(model_file)
