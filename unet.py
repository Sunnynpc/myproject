import torch
from torch import nn
from torch.nn.functional import interpolate

# 构建网络的block
class CNNLayer(nn.Module):
    # 传入输出通道和输出通道两个参数
    def __init__(self,C_in,C_out):
        super(CNNLayer, self).__init__()
        # 构建卷积块
        self.layer = nn.Sequential(
            # 定义一个卷积层，输入通道为c_in,输出通道为c_out，3*3的卷积核，步长为1，padding为1，padding模式为reflect
            nn.Conv2d(C_in,C_out,3,1,1,padding_mode="reflect"),
            # 归一化
            nn.BatchNorm2d(C_out),
            # 激活
            nn.LeakyReLU(),
            # 随机抑制当前层30%的神经元
            nn.Dropout2d(0.3),
            # 定义一个卷积层，输出通道与输入通道相同，3*3的卷积核，步长为1，padding为1，padding模式为reflect
            nn.Conv2d(C_out, C_out, 3, 1, 1, padding_mode="reflect"),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU(),
            nn.Dropout2d(0.4),
        )
    # 前向计算
    def forward(self,x):
        # 参数x传入卷积块得到输出并返回
        return self.layer(x)
# 下采样模块
class DownSampling(nn.Module):
    # 传入参数C
    def __init__(self,C):
        super(DownSampling, self).__init__()
        # 使用卷积进行下采样
        self.layer = nn.Sequential(
            # 定义一个卷积层，输入通道与输出通道不变，3*3的卷积核，步长为2，padding为1，padding模式为reflect
            nn.Conv2d(C,C,3,2,1,padding_mode="reflect"),
            # 激活
            nn.LeakyReLU(),
            # 归一化
            nn.BatchNorm2d(C)
        )
    #前向计算
    def forward(self,x):
        # 参数x传入下采样模块得到输出并返回
        return self.layer(x)

# 上采样模块
class UpSampling(nn.Module):
    # 传入参数C
    def __init__(self,C):
        super(UpSampling, self).__init__()
        # 使用卷积减小通道
        self.layer = nn.Sequential(
            # 定义一个卷积层，输入通道为C，输出通道为C/2，3*3的卷积核，步长为1，padding为1，padding模式为reflect
            nn.Conv2d(C,C//2,3,1,1,padding_mode="reflect"),
            # 激活
            nn.LeakyReLU(),
            # 归一化
            nn.BatchNorm2d(C//2)
        )
    # 前向计算，传入参数x和block的输出r
    def forward(self,x,r):
        # 将传入的x进行上采样，放大2倍
        up = interpolate(x,scale_factor=2,mode="nearest")
        # 将上采样后的结果传入卷积块得到结果
        x = self.layer(up)
        # 将结果与block的输出拼接到一起并返回
        return torch.cat((x,r),1)

# 构造unet网络
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 第一层CNNLayer卷积
        self.C1 = CNNLayer(3,64)
        # 第一层下采样
        self.D1 = DownSampling(64)
        # 第二层CNNLayer卷积
        self.C2 = CNNLayer(64,128)
        # 第二层下采样
        self.D2 = DownSampling(128)
        # 第三层CNNLayer卷积
        self.C3 = CNNLayer(128,256)
        # 第三层下采样
        self.D3 = DownSampling(256)
        # 第四层CNNLayer卷积
        self.C4 = CNNLayer(256,512)
        # 第四层下采样
        self.D4 = DownSampling(512)
        # 第五层CNNLayer卷积
        self.C5 = CNNLayer(512,1024)
        # 第一层上采样
        self.U1 = UpSampling(1024)
        # 第一层UpSampling卷积
        self.C6 = CNNLayer(1024,512)
        # 第二层上采样
        self.U2 = UpSampling(512)
        # 第二层UpSampling卷积
        self.C7 = CNNLayer(512,256)
        # 第三层上采样
        self.U3 = UpSampling(256)
        # 第三层UpSampling卷积
        self.C8 = CNNLayer(256,128)
        # 第四层上采样
        self.U4 = UpSampling(128)
        # 第四层UpSampling卷积
        self.C9 = CNNLayer(128,64)
        # 输出卷积层
        self.pre = nn.Conv2d(64,3,3,1,1)
    # 前向计算
    def forward(self,x):
        # 将x传入第一层CNNLayer卷积得到R1
        R1 = self.C1(x)
        # 将R1下采样后传入第二层卷积得到R2
        R2 = self.C2(self.D1(R1))
        # 将R2下采样后传入第三层卷积得到R3
        R3 = self.C3(self.D2(R2))
        # 将R3下采样后传入第四层卷积得到R4
        R4 = self.C4(self.D3(R3))
        # 将R4下采样后传入第五层卷积得到Y1
        Y1 = self.C5(self.D4(R4))
        # 将Y1进行上采样后传入第一层UpSampling卷积，与R4进行拼接后得到O1
        O1 = self.C6(self.U1(Y1,R4))
        # 将O1进行上采样后传入第一层UpSampling卷积，与R3进行拼接后得到O2
        O2 = self.C7(self.U2(O1, R3))
        # 将O2进行上采样后传入第一层UpSampling卷积，与R2进行拼接后得到O3
        O3 = self.C8(self.U3(O2, R2))
        # 将O3进行上采样后传入第一层UpSampling卷积，与R1进行拼接后得到O4
        O4 = self.C9(self.U4(O3, R1))
        # 将O4传入输出层卷积得到输出并返回
        return self.pre(O4)
# 主函数、测试
if __name__ == '__main__':
    # 随机一个形状为【2，3，256，256】的数
    x = torch.randn(2,3,256,256)
    # 实例化网络
    net = UNet()
    # 将x传入模型得到输出
    out = net(x)
    # 输出out的形状
    print(out.shape)