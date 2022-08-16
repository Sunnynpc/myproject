import torch
from torch import nn

class Arcsoftmax(nn.Module):
    def __init__(self,feature_nums,cls_nums):
        super(Arcsoftmax, self).__init__()
        self.w = nn.Parameter(torch.randn(feature_nums,cls_nums))

    def forward(self,x,s=10,m=1):
        x_norm = nn.functional.normalize(x, dim=1)
        w_norm = nn.functional.normalize(self.w, dim=0)
        cosa = torch.matmul(x_norm, w_norm)/10
        a = torch.acos(cosa)
        top = torch.exp(s*torch.cos(a+m))
        down = top + torch.sum(torch.exp(s*cosa), dim=1, keepdim=True)-torch.exp(s*cosa)
        arcsoftmax = top/down
        return torch.log(arcsoftmax)



if __name__ == '__main__':
    arc = Arcsoftmax(128,3488)
    data = torch.randn(1,128)
    out =arc(data)
    print(data)
    print(out)