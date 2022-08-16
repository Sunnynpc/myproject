import torch
from torch import nn
from torch.nn.functional import interpolate

# 定义一个卷积，卷积核为3*3，步长为1
layer = nn.Conv2d(1,1,3,1,padding=1)
x = torch.randn(1,1,3,3)
# 将x卷积得到结果
out = layer(x)
# 打印结果形状
print(out.shape)
# 打印结果
print(out)
# 插值法，将图片大小放大到原来的2倍
out = interpolate(out, scale_factor=2, mode='nearest')
print(out.shape)
print(out)