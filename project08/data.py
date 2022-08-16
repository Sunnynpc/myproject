from docx import Document
from sklearn.model_selection import train_test_split


# def get_paragraphs(docx_path):
#     document = Document(docx_path)
#
#     all_paragraphs = document.paragraphs
#
#     paragraphs_texts = []
#     for paragraph in all_paragraphs:
#         paragraphs_texts.append(paragraph.text)
#
#     return paragraphs_texts
#
# docx_path = 'F:\data\能降解PET的酶.docx'
# paragrapgh_texts = get_paragraphs(docx_path)
# with open(docx_path[:-4]+'txt','w',encoding='UTF-8') as file:
#     for item in paragrapgh_texts:
#         if not item or len(item)<5:
#             continue
#         file.write(item.replace('\n','')+'\n')
positive_path = r'F:\data\1.txt'
positive_file = open(positive_path,'r+')
positive_data = positive_file.readlines()
train_set, test_set = train_test_split(positive_data,test_size=0.2,random_state=42)
with open(r'F:\data\train\1.txt','w') as file1:
    for item in train_set:
        file1.write(item)
with open(r'F:\data\test\1.txt','w') as file2:
    for item in test_set:
        file2.write(item)