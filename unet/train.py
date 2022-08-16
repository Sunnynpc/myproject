import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from unet import UNet
from MKData import MKDataset
from torchvision.utils import save_image
# 定义数据集的路径
path = r"F:\ VOCdevkit\VOCdevkit\VOC2012"
# 定义模型的保存位置
module = r"params/mode.pth"
# 定义图片的保存目录
img_save_path = r"f://train_img"
# 实例化网络，放到cuda上训练
net = UNet().cuda()
# 定义网络优化器
opt = torch.optim.Adam(net.parameters())
# loss_func = nn.CrossEntropyLoss()
# 均方差损失函数
loss_func = nn.MSELoss()
# 加载数据集
data_laoder = DataLoader(MKDataset(path), batch_size=2, shuffle=True)
# 判断模型是否存在
if os.path.exists(module):
    # 存在时加载参数
    net.load_state_dict(torch.load(module))
else:
    print("No Params!")
# 判断图片存储路径是否存在
if not os.path.exists(img_save_path):
    # 不存在时创建一个文件夹
    os.makedirs(img_save_path)

# 训练模型
epoch = 1
# 一直训练
while True:
    # 取出数据集中的数据
    for i, (xs, ys) in enumerate(data_laoder):
        # 把数据放到cuda上
        xs = xs.cuda()
        ys = ys.cuda()
        # 将数据放入模型得到输出
        xs_ = net(xs)
        # print(xs_.shape)
        # print(ys.shape)
        # 根据输出计算损失
        loss = loss_func(xs_,ys)
        # 清空梯度
        opt.zero_grad()
        # 自动求导
        loss.backward()
        # 更新梯度
        opt.step()
        # 每隔5个批次
        if i%5 == 0:
            # 打印轮次、批次、损失
            print("epoch:{},count:{},loss:{}".format(epoch, i, loss.item()))
    # 每个批次取一张图片
    x = xs[0]
    # 每个批次取一张模型输出图片
    x_ = xs_[0]
    # 每个批次取一张标签图片
    y = ys[0]
    # 将三张图片在0维拼接成一张
    img = torch.stack([x, x_, y], 0)
    # 保存图片
    save_image(img.cpu(), os.path.join(img_save_path, "{}.png".format(epoch)))
    # 轮次+1
    epoch +=1
    # 保存模型参数
    torch.save(net.state_dict, module)
    print("参数保存成功！")

