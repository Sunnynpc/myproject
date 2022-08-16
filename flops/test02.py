import torch
from torchvision import models
import thop
from thop import clever_format
# 实例化mobilenet_v2模型
net = models.mobilenet_v2()
# 主函数，测试
if __name__ == '__main__':
    x = torch.randn(1,3,224,224)
    # 得到网络模型的参数量、计算量
    flops,params = clever_format(thop.profile(net,(x,)))
    # 输出模型参数量、计算量
    print(flops, params)