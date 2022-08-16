import jieba
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer

path_txt = './test2.txt'   #文档在电脑上所在位置
txt = open(path_txt,'r',encoding='utf-8').read()
excludes = ["，","：","“","。","”","、","；"," ","\n","…","‘","’","（","）"]
# words = jieba.lcut(txt)
# counts = {}
# for word in words:
#     if word in excludes or word.isdigit():
#         continue
#     else:
#         counts[word] = counts.get(word,0) + 1
# items = list(counts.items())
# items.sort(key=lambda x:x[1], reverse=True)
# # for i in range(len(items)):
# #     word, count =items[i]
# #     print("{0}{1}".format(word, count))
# print(counts)
# # print(len(items))
contents = [txt]
tf = CountVectorizer(tokenizer=jieba.lcut, stop_words=excludes)
res1 = tf.fit_transform(contents)        # 使用函数拟合转置contents
print(tf.vocabulary_)
print(len(tf.vocabulary_))
