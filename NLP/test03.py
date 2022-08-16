import re

import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer

# path_txt = './test2.txt'
# txt = open(path_txt,'r',encoding='utf-8').read()
excel = pd.read_excel(r'./data/t.xlsx',usecols="B,C")
excel_l = excel.values.tolist()

excludes = ['μ', '〔', '\t', '﹪', 'Ｉ', '?', 'ｋ', 'Ｕ', '【', 'Ⅶ', 'ｌ', '～',
            '(', '\n', '‘', '&', 'ｂ', 'ω', '#', ';', '+', 'ｍ', '$', ')', '；',
            '。', '▼', 'ａ', '’', '﹚', '→', '》', 'Ａ', '③', '■', ' ', 'α', 'Ⅱ',
            '.', '，', 'Ｑ', '﹙', '：', 'Π', '“', '[', 'Ⅸ', 'Ｄ', '!', '′',
            '\u3000', 'Ⅴ', '＼', ',', '=', '□', '☆', '★', 'ｓ', '@', '\ue2d9',
            '、', '”', '}', 'Ｂ', '】', '—', '②', '！', '］', 'ｕ', '○', '*', 'Ｃ',
            '（', '×', '/', 'Δ', '▲', 'γ', ':', 'Ⅳ', '{', 'Ｋ', '＿', 'Ｓ', '~',
            '％', '⑾', '《', 'Ⅲ', '>', '-', '^', ']', '◆', '㊣', '？', '●', '≧',
            'Ｇ', '<', 'β', '·', 'а', '△', 'ｇ', '\\', '◇', '－', '%', "'", '※',
            'Ⅰ', '）', '〕', '\x7f', '①', '≦']
char_dic = ["'", '，', ')', '(', '[', ':', '（', ',', 'β', 'Ⅸ', 'α', '）', '-', 'Ⅱ', ']', '、', '/', '.']
print(len(char_dic))
curr_words = []
data = []
for s in excel_l:
    for i in s:
        a = re.sub('[^\u4e00-\u9fa5]','',i)
        data.append(a)
# print(data)

for sen in data:
    for word in jieba.cut(sen):         # 遍历文章中的每个词并分词
        curr_words.append(word)
curr_words = list(set(curr_words))
curr_words.sort()
print(curr_words)
print(len(curr_words))