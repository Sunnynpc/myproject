import torch

a = torch.randn(10,3,requires_grad=True)
# a.requires_grad=True
# print(a.requires_grad)

print(a)
print(a[[1,3,5]])
