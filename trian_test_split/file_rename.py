import os
import sys

root = r'E:\PycharmProjects\yolo\data\imgs'

filelist = os.listdir(root)
# 得到进程当前工作目录
currentpath = os.getcwd()
#将当前工作目录修改为待修改文件夹的位置
os.chdir(root)
for i,filename in enumerate(filelist):
    os.rename(filename,str(i+601).zfill(4)+'.jpg')
    # os.rename(filename, str(i ).zfill(4) + '.jpg')

os.chdir(currentpath)       #改回程序运行前的工作目录
sys.stdin.flush()     #刷新
print('修改成功')