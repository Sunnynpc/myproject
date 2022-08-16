import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# 编码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 构建一个全连接神经网络提取特征，输出为10
        self.layer = nn.Sequential(
            nn.Linear(784,512,bias=False),
            # 特征归一化
            nn.BatchNorm1d(512),
            # 激活
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10, bias=False),
            nn.BatchNorm1d(10),
            nn.ReLU()
        )
    # 前向计算
    def forward(self,x):
        # 将x传入编码器得到输出并返回
        return self.layer(x)
# 解码器
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # 构建一个全连接网络还原图片数据
        self.layer = nn.Sequential(
            nn.Linear(10, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 784, bias=False),
            nn.BatchNorm1d(784),
            nn.ReLU()
        )

    # 前向计算
    def forward(self, x):
        # 将x传入解码器得到输出并返回
        return self.layer(x)
# 构造网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 实例化编码器
        self.encoder = Encoder()
        # 实例化解码器
        self.decoder = Decoder()
    # 前向计算
    def forward(self,x):
        # 将x传入模型进行编码
        encoder_out = self.encoder(x)
        # 将编码结果传入解码器得到结果
        out = self.decoder(encoder_out)
        # 将结果返回
        return out
# 主函数，测试
if __name__ == '__main__':
    # 将网络放到cuda上训练
    net = Net().cuda()
    # 加载手写数字数据集
    mnist_data = datasets.MNIST(root="E:\data\MNIST_data",train=True,transform=transforms.ToTensor(),download=True)
    train_laoder = DataLoader(mnist_data, batch_size=100, shuffle=True)
    # 交叉熵损失函数
    loss_func = nn.MSELoss()
    # 模型优化器
    opt = torch.optim.Adam(net.parameters())
    k = 0
    # 训练轮次
    for epoch in range(100000):
        # 将数据从数据集取出
        for i,(img,_) in enumerate(train_laoder):
            # 更改图片形状，放到cuda上
            img = img.reshape(-1,28*28).cuda()
            # 将数据传入模型得到输出
            out = net(img)
            # 计算损失
            loss = loss_func(out,img)
            # 清空梯度
            opt.zero_grad()
            # 自动求导
            loss.backward()
            # 更新梯度
            opt.step()

            # 批次为10，输出loss
            if i % 10 == 0:
                print(loss.item())
                # 解绑
                fack_img = out.detach()
                # 还原图片形状
                img = img.reshape(-1,1,28,28)
                # 将生成图片形状改变成和原来图片一样
                fack_img = fack_img.reshape(-1,1,28,28)
                # 保存生成的图片，把tensor保存成PIL，10张拼接成1张
                save_image(fack_img,"img/{}-fack_img.png".format(k), nrow=10)
                save_image(img, "img/{}-real_img.png".format(k), nrow=10)
                k+=1