import torch
# 创建一个张量
# a = torch.tensor([12,34,21,56,78,93])、
# 定义一个掩码
# mask1 = torch.tensor([True,True,False,True,False,False])
# mask2 = torch.tensor([1,1,0,1,0,0])
# 打印掩码后的结果
# print(a[mask1])
# print(a*mask2)
# 创建一个随机数a，形状为【1，2，3，3】
a = torch.randn(1,2,3,3)
# 创建一个随机数b，形状为【1，2，3，3】
b = torch.randn(1,2,3,3)
# 将a与b在第一维度上进行拼接得到d
d = torch.cat((a,b),dim=1)
# a与b相加得到e
e = a+b
# 打印d的形状
print(d.shape)
# 打印e的形状
print(e.shape)