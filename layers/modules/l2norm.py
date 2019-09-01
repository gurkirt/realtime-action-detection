
import torch.nn as nn
import torch
import torch.nn.functional as F

# class L2Norm(nn.Module):
#     def __init__(self,n_channels, scale):
#         super(L2Norm,self).__init__()
#         self.n_channels = n_channels
#         self.gamma = scale or None
#         self.eps = 1e-10
#         self.weight = nn.Parameter(torch.Tensor(self.n_channels))
#         self.reset_parameters()

#     def reset_parameters(self):
#         init.constant(self.weight,self.gamma)

#     def forward(self, x):
#         norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
#         x /= norm
#         out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
#         return out

class L2Norm(nn.Module):
    def __init__(self, in_channels, initial_scale):
        super(L2Norm, self).__init__()
        self.in_channels = in_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels))
        self.initial_scale = initial_scale
        self.reset_parameters()

    def forward(self, x):
        return (F.normalize(x, p=2, dim=1)
                * self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3))

    def reset_parameters(self):
        self.weight.data.fill_(self.initial_scale)