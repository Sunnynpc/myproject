import torch

# a = torch.tensor([[1, 2, 3, 4], [6, 7, 8, 9]])
# b = torch.tile(a[:, None], (1, a.shape[-1], 1))
# print(b)
# I = torch.eye(b.shape[-2], b.shape[-1])
# I = torch.tile(I[None], (b.shape[0], 1, 1))
# print(I)
#
# b_i = b * I
# b_sum = b - b_i
#
# print(b_i)
# print(b_sum)
# print(b_i.sum(-1) / (b_sum.sum(-1) + b_i.sum(-1)))

a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
b = torch.diag_embed(a)
# c = torch.tile(a[:, None], (1, a.shape[-1], 1))

d = c - b
print(c.sum(-1)/(d.sum(-1)+c.sum(-1)))

