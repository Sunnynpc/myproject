import torch
from torch import nn
# 构造模型需要的参数：通道扩充倍数、输出通道、模型调用次数，最后一次的步长
config = [
    [-1,32,1,2],
    [1,16,1,1],
    [6,24,2,2],
    [6,32,3,2],
    [6,64,4,2],
    [6,96,3,1],
    [6,160,3,2],
    [6,320,1,1]
]
# 构造残差块
class Block(nn.Module):
    # 传入参数：输入通道、重复次数、通道扩充倍数、输出通道、模型调用次数，最后一次的步长
    def __init__(self,p_c,i,t,c,n,s):
        super(Block, self).__init__()
        # 每个重复的最后一次负责下采样和通道处理，所以i=n-1的时候进行操作
        self.i = i
        self.n = n
        # 判断是否是最后一次重复，最后一次步长为s
        _s = s if i == n-1 else 1
        # 判断是否是最后一次重复，最后一次负责将通道变换为下层的输入
        _c = c if i == n-1 else p_c

        _p_c = p_c * t# 输入通道的扩增倍数
        # 构造layer
        self.layer = nn.Sequential(
            # 第一层卷积，扩增通道，卷积核为1*1
            nn.Conv2d(p_c, _p_c, 1, 1, bias=False),
            # 特征归一化
            nn.BatchNorm2d(_p_c),
            # 激活
            nn.ReLU6(),
            # 第二次卷积，深度可分离卷积，卷积核为3*3
            nn.Conv2d(_p_c, _p_c, 3, _s, padding=1, groups=_p_c, bias=False),
            nn.BatchNorm2d(_p_c),
            nn.ReLU6(),
            # 第三次卷积，减少通道，卷积核为1*1
            nn.Conv2d(_p_c, _c, 1, 1, bias=False),
            nn.BatchNorm2d(_c)
        )
    # 前向计算
    def forward(self,x):
        # 判断是否最后一次重复，最后一次不做残差
        if self.i == self.n-1:
            return self.layer(x)
        else:
            return self.layer(x) + x

# 构造模型
class MobileNet_v2(nn.Module):
    # 构造函数,将构造模型需要的参数传入
    def __init__(self,config):
        super(MobileNet_v2, self).__init__()
        # 实例化参数
        self.config = config
        # 输入的第一层
        self.input_layer = nn.Sequential(
            # 第一层卷积，3*3的卷积核
            nn.Conv2d(3,32,3,2,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )
        # 实例化变量存放构建的模型
        self.blocks = []
        # 从参数中拿出输入的通道
        p_c = self.config[0][1]
        # 将通道扩充倍数、输出通道、模型调用次数，最后一次的步长从参数中拿出
        for t,c,n,s in self.config[1:]:
            # 循环调用Block
            for i in range(n):
                # 将参数传入Block，构建模型
                self.blocks.append(Block(p_c,i,t,c,n,s))
            # 每次循环后修改输入通道为上一次的输出通道
            p_c = c
        # 将构建的模型存放到隐藏层中
        self.hidden_layer = nn.Sequential(*self.blocks)# 可变参数
        # 构建输出层
        self.output_layer = nn.Sequential(
            # 第一层卷积，1*1的核
            nn.Conv2d(320,1280,1,1,bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(),
            # 平均池化，7*7的窗口
            nn.AvgPool2d(7,1),
            # 第2层卷积
            nn.Conv2d(1280,10,1,1)
        )
    # 前向计算
    def forward(self,x):
        # 将x放入输入层得到结果
        h = self.input_layer(x)
        # 将上一次的结果放入隐藏层得到结果
        h = self.hidden_layer(h)
        # 将隐藏层的结果放入输出层得到结果
        h = self.output_layer(h)
        # 将最后的结果返回
        return h

#主函数，测试
if __name__ == '__main__':
    # 实例化网络
    net = MobileNet_v2(config)
    x = torch.randn(1,3,224,224)
    # 将x传入模型
    y = net(x)
    # 打印结果形状
    print(y.shape)
    # 打印网络
    print(net)