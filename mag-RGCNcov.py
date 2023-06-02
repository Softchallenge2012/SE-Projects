#!/usr/bin/env python
# coding: utf-8

# In[1]:


from copy import copy
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.data import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import pandas as pd
from logger import logger


from collections import Counter
import numpy as np


# In[2]:


import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')


# In[4]:


logger = Logger(runs)


# In[ ]:





# In[5]:


import copy
import argparse

import torch
import torch.nn.functional as F
from torch.nn import Parameter, ModuleDict, ModuleList, Linear, ParameterDict
from torch_sparse import SparseTensor

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

#from logger import Logger


class RGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, node_types, edge_types):
        super(RGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # `ModuleDict` does not allow tuples :(
        self.rel_lins = ModuleDict({
            f'{key[0]}_{key[1]}_{key[2]}': Linear(in_channels, out_channels,
                                                  bias=False)
            for key in edge_types
        })

        self.root_lins = ModuleDict({
            key: Linear(in_channels, out_channels, bias=True)
            for key in node_types
        })

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins.values():
            lin.reset_parameters()
        for lin in self.root_lins.values():
            lin.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        out_dict = {}
        for key, x in x_dict.items():
            out_dict[key] = self.root_lins[key](x)

        for key, adj_t in adj_t_dict.items():
            key_str = f'{key[0]}_{key[1]}_{key[2]}'
            x = x_dict[key[0]]
            out = self.rel_lins[key_str](adj_t.matmul(x, reduce='mean'))
            out_dict[key[2]].add_(out)

        return out_dict


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, edge_types):
        super(RGCN, self).__init__()

        node_types = list(num_nodes_dict.keys())

        self.embs = ParameterDict({
            key: Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        self.convs = ModuleList()
        self.convs.append(
            RGCNConv(in_channels, hidden_channels, node_types, edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, node_types,
                         edge_types))
        self.convs.append(
            RGCNConv(hidden_channels, out_channels, node_types, edge_types))

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embs.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        x_dict = copy.copy(x_dict)
        for key, emb in self.embs.items():
            x_dict[key] = emb

        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, adj_t_dict)
            for key, x in x_dict.items():
                x_dict[key] = F.relu(x)
                x_dict[key] = F.dropout(x, p=self.dropout,
                                        training=self.training)
        return self.convs[-1](x_dict, adj_t_dict)


def train(model, x_dict, adj_t_dict, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x_dict, adj_t_dict)['paper'].log_softmax(dim=-1)
    loss = F.nll_loss(out[train_idx], y_true[train_idx].squeeze())
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x_dict, adj_t_dict, y_true, split_idx, evaluator):
    model.eval()

    out = model(x_dict, adj_t_dict)['paper']
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc


# In[6]:



device=0
log_steps=1
num_layers=2
hidden_channels=64
dropout=0.5
lr=0.01
epochs=5
runs=2

logger = Logger(runs)
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

dataset = PygNodePropPredDataset(name='ogbn-mag', root='data')
split_idx = dataset.get_idx_split()
data = dataset[0]

# We do not consider those attributes for now.
data.node_year_dict = None
data.edge_reltype_dict = None

print(data)

# Convert to new transposed `SparseTensor` format and add reverse edges.
data.adj_t_dict = {}
for keys, (row, col) in data.edge_index_dict.items():
    sizes = (data.num_nodes_dict[keys[0]], data.num_nodes_dict[keys[2]])
    adj = SparseTensor(row=row, col=col, sparse_sizes=sizes)
    # adj = SparseTensor(row=row, col=col)[:sizes[0], :sizes[1]] # TEST
    if keys[0] != keys[2]:
        data.adj_t_dict[keys] = adj.t()
        data.adj_t_dict[(keys[2], 'to', keys[0])] = adj
    else:
        data.adj_t_dict[keys] = adj.to_symmetric()
data.edge_index_dict = None

x_types = list(data.x_dict.keys())
edge_types = list(data.adj_t_dict.keys())

model = RGCN(data.x_dict['paper'].size(-1), hidden_channels,
             dataset.num_classes, num_layers, dropout,
             data.num_nodes_dict, x_types, edge_types)

data = data.to(device)
model = model.to(device)
train_idx = split_idx['train']['paper'].to(device)

evaluator = Evaluator(name='ogbn-mag')
logger = Logger(runs)

for run in range(runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, 1 + epochs):
        loss = train(model, data.x_dict, data.adj_t_dict,
                     data.y_dict['paper'], train_idx, optimizer)
        result = test(model, data.x_dict, data.adj_t_dict,
                      data.y_dict['paper'], split_idx, evaluator)
        logger.add_result(run, result)

        if epoch % log_steps == 0:
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')

    logger.print_statistics(run)
logger.print_statistics()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




