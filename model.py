import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ltlf2dfa.parser.ltlf import LTLfAnd, LTLfUntil, LTLfNot, LTLfAlways, LTLfAtomic, LTLfNext, LTLfOr, LTLfEventually
from layers import opLayer, invLayer

invOp = [LTLfAnd, LTLfOr]
binOp = [LTLfUntil]
uOp = [LTLfNot, LTLfNext, LTLfEventually, LTLfAlways]


class LTLfNet(nn.Module):
    def __init__(self, device, var_num=1000, binop_num=1, invop_num=2,
                 uop_num=4, num_classes=2):
        super(LTLfNet, self).__init__()
        self.device = device
        self.embed_dim = var_num + 1
        self.num_classes = num_classes
        self.var_num = var_num
        # node init embedding

        self.var_dict = {}
        self.invOp_layers = nn.ModuleList([invLayer(self.embed_dim) for _ in range(invop_num)])
        self.binop_layers = nn.ModuleList([opLayer(2, self.embed_dim) for _ in range(binop_num)])
        self.uop_layers = nn.ModuleList([opLayer(1, self.embed_dim) for _ in range(uop_num)])
        # MLP
        self.outLayer = nn.Linear(self.embed_dim, num_classes)
        # for var indexing
        self.var_idx = torch.randperm(var_num) + 1

    def init_index(self):
        self.var_dict.clear()
        self.var_idx = torch.randperm(self.var_num) + 1

    def embed_atom(self, a: str) -> Tensor:
        vd = self.var_dict
        if a not in vd:
            if len(vd) == self.var_num:
                idx = 0     # for unk
            else:
                vd[a] = idx = self.var_idx[len(vd)]
            # endif
        else:
            idx = vd[a]
        # end if
        one_hot = torch.zeros((1, self.embed_dim)).to(self.device)
        one_hot[0, idx] = 1
        return one_hot

    def embed_node(self, f) -> Tensor:
        # atom
        if isinstance(f, LTLfAtomic):
            return self.embed_atom(f.s)
        # endif
        # invOp
        for i, t in enumerate(invOp):
            if isinstance(f, t):
                layer = self.invOp_layers[i]
                data = None
                for sn in f.formulas:
                    if data is None:
                        data = self.embed_node(sn)
                    else:
                        data = torch.cat([data, self.embed_node(sn)])
                # endfor
                return layer(data)
            # endif
        # endfor
        # binary
        for i, t in enumerate(binOp):
            if isinstance(f, t):
                layer = self.binop_layers[i]
                data = [self.embed_node(sn) for sn in f.formulas]
                return layer(data)
            # endif
        # endfor
        # unary
        for i, t in enumerate(uOp):
            if isinstance(f, t):
                layer = self.uop_layers[i]
                data = [self.embed_node(f.f)]
                return layer(data)
            # endif
        # endfor
        # unk type
        raise ValueError(f"Unknown arg: {f}. Type: {type(f)}.")

    def forward(self, node_list: list, bn=None) -> Tensor:
        # init var_index_dict
        self.init_index()
        # forward
        if bn is None:
            x = self.embed_node(node_list[0])
        else:
            bs = len(bn)
            x = torch.zeros((bs, self.embed_dim)).to(self.device)
            for i in range(bs):
                x[i] = self.embed_node(node_list[bn[i]])
            # endfor
        # endif
        x = F.relu(x)
        x = self.outLayer(x)
        self.init_index()
        return F.log_softmax(x, dim=1)
        

if __name__ == '__main__':
    pass