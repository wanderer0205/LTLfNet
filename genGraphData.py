import pandas as pd
from ltlf2dfa.parser.ltlf import LTLfParser, LTLfAnd, LTLfUntil, LTLfNot, LTLfAlways, LTLfAtomic, LTLfNext, LTLfOr, LTLfEventually, LTLfImplies, LTLfRelease
import scipy.sparse as sp
from torch import LongTensor
from torch.utils.data import Dataset
from tqdm import trange

# ltlParser
parser = LTLfParser()

def getNodes(f, nodes: dict):
    self_formula = f
    if self_formula not in nodes:
        nodes[self_formula] = len(nodes)
    # endif
    if isinstance(f, LTLfAtomic):
        pass
    elif isinstance(f, LTLfAnd) or isinstance(f, LTLfUntil) \
            or isinstance(f, LTLfOr) or isinstance(f, LTLfRelease)\
            or isinstance(f, LTLfImplies):
        for formula in f.formulas:
            getNodes(formula, nodes)
    elif isinstance(f, LTLfNot) or isinstance(f, LTLfNext) \
            or isinstance(f, LTLfAlways) or isinstance(f, LTLfEventually):
        getNodes(f.f, nodes)
    else:
        raise ValueError(f"Unknown arg: {f}.")
    # endif

# nodes: raw -> dup with X
def getGraph(nodes: dict, next_edge=False, undirected=False) -> (sp.coo_matrix, dict):

    def getNext(f):
        return parser(f"X({str(f)})")

    from_e, to_e = [], []

    def add_edge(fr, to):
        from_e.append(fr)
        to_e.append(to)
        if undirected:
            to_e.append(fr)
            from_e.append(to)
        # endif

    n = len(nodes)  # total node num
    next_nodes = {}
    for subf in nodes:
        next_subf = getNext(subf)
        next_nodes[next_subf] = nodes[subf] + n
        if next_edge:
            add_edge(nodes[subf], nodes[subf] + n)
    # endfor

    for f in nodes:
        if isinstance(f, LTLfNot):
            # fi = !fj => fi = !fj
            fj = f.f
            add_edge(nodes[fj], nodes[f])
        elif isinstance(f, LTLfNext):
            # fi = Xfj => fi = Xfj
            # fi = Xfj
            add_edge(next_nodes[f], nodes[f])
        elif isinstance(f, LTLfEventually) or isinstance(f, LTLfAlways):
            # fi = F fj => fi = fj | X fi
            # fi = G fj => fi = fj & X fi
            fj = f.f
            add_edge(nodes[fj], nodes[f])
            add_edge(next_nodes[getNext(f)], nodes[f])
        elif isinstance(f, LTLfAnd) or isinstance(f, LTLfOr):
            # the same
            for fj in f.formulas:
                add_edge(nodes[fj], nodes[f])
            # endfor
        elif isinstance(f, LTLfUntil):
            # fi = fj U fk => fi = fk | (fj & Xfi)
            add_edge(next_nodes[getNext(f)], nodes[f])
            for fj in f.formulas:
                add_edge(nodes[fj], nodes[f])
            # endfor
        elif isinstance(f, LTLfAtomic):
            pass
        else:
            raise ValueError(f"Unknown arg: {f}.\tType: {type(f)}.")
        # endif
    # endfor
    edge_index = [from_e, to_e]
    node_list = []
    for k in nodes:
        node_list.append(k)
    for k in next_nodes:
        node_list.append(k)

    return node_list, edge_index
    # return next_nodes, sp.coo_matrix(adj), edge_index


class TLDataSet(Dataset):
    def __init__(self, path, next_edge=False, undirected=False):
        self.path = path
        self.data = []
        df = pd.read_json(path)
        self.processDf(df, next_edge, undirected)

    def processDf(self, df, next_edge, undirected):
        print(f"Processing data from {self.path}.")
        for i in trange(len(df), ncols=80, desc=f'Processing'):
            data = df.loc[i]
            f_raw, y = data['inorder'], data['issat']
            formula = parser(f_raw)
            node_dict = {}
            getNodes(formula, node_dict)
            node_list, edge_index = getGraph(node_dict, next_edge, undirected)
            self.data.append((node_list, edge_index, y))
            # next_dict, adj, edge_index = getGraph(node_dict)
            # self.data.append((node_dict, next_dict, edge_index, y))
        # endfor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def collate(batch):
    node_list, y_list = [], []
    bat_index = [] 
    e_index = [[], []]
    pos = 0   
    for nl, edge_index, y in batch:
        bat_index.append(pos) 
        n = len(nl)     
        for e, ae in zip(edge_index, e_index):
            for i in e:
                ae.append(i + pos)    # edge_index
            # endfor
        # endfor
        node_list.extend(nl)
        y_list.append(int(y))
        pos += n 
    # endfor
    return node_list, LongTensor(e_index), LongTensor(y_list), LongTensor(bat_index)


if __name__ == '__main__':
    formula = parser('b U X !c')
