import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision
import os
from PIL import Image
from torchvision.utils import save_image

# 归一化，将图片转化为tensor
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 数据集处理
class MKDataset(Dataset):
    # 初始化数据，将目录path传入
    def __init__(self,path):
        # 实例化路径
        self.path = path
        # 遍历目录下的图片得到数据集
        self.name = os.listdir(os.path.join(path,"SegmentationClass"))
    # 返回数据集长度
    def __len__(self):
        return len(self.name)
    # 对单条数据进行处理，将索引index传入
    def __getitem__(self, index):
        # 定义数据和标签的背景图片
        black1 = torchvision.transforms.ToPILImage()(torch.zeros(3,256,256))
        black0 = torchvision.transforms.ToPILImage()(torch.zeros(3, 256, 256))
        # 索引数据，得到标签图片名称
        name = self.name[index]
        # 对名称进行切片，加上jpg得到图片名称
        name_jpg = name[:-3]+"jpg"
        # 得到图片目录
        img1_path = os.path.join(self.path, "JPEGImages")
        # 得到标签目录
        img0_path = os.path.join(self.path, "SegmentationClass")
        # 得到图片路径，根据图片路径得到图片数据
        img1 = Image.open(os.path.join(img1_path,name_jpg))
        # 得到标签路径，根据标签路径得到标签数据
        img0 = Image.open(os.path.join(img0_path,name))

        # 得到图片的宽和高
        img1_size = torch.Tensor(img1.size)
        # 得到图片宽高中更长的那条边
        l_max_index = img1_size.argmax()
        # 得到图片缩放比例
        ratio = 256/img1_size[l_max_index]
        # 得到对图片进行缩放后的图片尺寸
        img1_resize = img1_size * ratio
        # 将类型转换为长整型
        img1_resize = img1_resize.long()
        # 对图片进行缩放
        img1_use = img1.resize(img1_resize)
        img0_use = img0.resize(img1_resize)
        # 将图片粘贴到黑色背景上
        black0.paste(img0_use, (0,0))
        black1.paste(img1_use, (0,0))
        # 将归一化后的数据与标签返回
        return transform(black1),transform(black0)
# 主函数、测试
if __name__ == '__main__':
    # 定义次数
    i=1
    # 实例化数据集类，将根目录传入数据集类
    dataset = MKDataset(r"D:\MyData\VOC 2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012")
    # 遍历数据集
    for a, b in dataset:
        # 打印次数
        print(i)
        # 打印数据形状
        print(a.shape)
        # 打印标签形状
        print(b.shape)
        # 保存图片，将tensor转为图片
        save_image(a,"data/{0}.jpg".format(i), nrow=1)
        save_image(b,"data/{0}.png".format(i), nrow=1)
        # 次数+1
        i+= 1