import torch
from torch.functional import F

# _out = torch.randn(3, 4)
#
# _out_i = torch.exp(_out)
# _out_sum = torch.sum(_out, dim=-1, keepdim=True)
# print(_out_i.shape, _out_sum.shape)
# print(_out_i / _out_sum)

f = torch.tensor([[1., 2, 3], [2, 3, 4]])
w = torch.tensor([[1., 2, 3], [2, 3, 4]])

f_n = F.normalize(f, dim=-1)
w_n = F.normalize(w, dim=-1)
#
# angle = torch.arccos(f_n * w_n)
#
# print(angle)
print(f_n)

# print(torch.eye(3,3))
a = torch.randn(21,1)
b = torch.randn(21,3844)
print(a*b)

