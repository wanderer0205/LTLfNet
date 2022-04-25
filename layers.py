import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class opLayer(nn.Module):
    def __init__(self, ary, e_dim=128):
        super(opLayer, self).__init__()
        # 2 layer mlp with residual
        self.input_layer = nn.Linear(ary * e_dim, e_dim, bias=False)
        self.residual_layer = nn.Linear(ary * e_dim, e_dim, bias=False)
        self.output_layer = nn.Linear(e_dim, e_dim, bias=True)
        # self.layers = nn.ModuleList([nn.Linear(e_dim, e_dim) for i in range(layer_num)])

    def forward(self, x):
        # x: [x1, x2 ...]
        l0 = torch.cat(x).reshape(1, -1)
        l1 = F.relu(self.input_layer(l0))
        lout = self.residual_layer(l0) + self.output_layer(l1)
        return F.normalize(lout, p=2, dim=1)

# for invariant op
class invLayer(nn.Module):
    def __init__(self, e_dim=128):
        super(invLayer, self).__init__()
        # 2 layer mlp with residual
        self.input_layer = nn.Linear(e_dim, e_dim, bias=False)
        self.residual_layer = nn.Linear(e_dim, e_dim, bias=False)
        self.output_layer = nn.Linear(e_dim, e_dim, bias=True)
        # self.layers = nn.ModuleList([nn.Linear(e_dim, e_dim) for i in range(layer_num)])

    def forward(self, x):
        # x: [x1, x2 ...]
        l0 = torch.mean(x, dim=0).unsqueeze(0)
        l1 = F.relu(self.input_layer(l0))
        lout = self.residual_layer(l0) + self.output_layer(l1)
        return F.normalize(lout, p=2, dim=1)
