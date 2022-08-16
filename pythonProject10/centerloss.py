import torch
import torch.nn as nn

class Centerloss(nn.Module):
    def __init__(self):
        super().__init__()
        # 类中心点
        self.center = nn.Parameter(torch.randn(10,2),requires_grad=True)
    # ys:标签，features:特征
    def forward(self,features,ys,lambdas=2):
        # 范数归一化
        # features = nn.functional.normalize(features)
        center_exp = self.center.index_select(dim=0,index=ys.long())
        # 统计各分类数量
        count = torch.histc(ys,bins=int(max(ys).item()+1),min=0,max=int(max(ys).item()))
        count_exp = count.index_select(dim=0,index=ys.long())
        loss = lambdas/2*torch.mean(torch.div(torch.sum(torch.pow(features-center_exp,2),dim=1),count_exp))
        return loss

if __name__ == '__main__':#测试
    a = Centerloss()
    feature = torch.randn(5, 2, dtype=torch.float32)
    ys = torch.tensor([0, 0, 1, 0, 1,], dtype=torch.float32)
    b = a(feature, ys)