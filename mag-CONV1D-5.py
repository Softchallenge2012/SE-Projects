#!/usr/bin/env python
# coding: utf-8

# In[2]:


from copy import copy
import argparse
from tqdm import tqdm

import os.path as osp

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.data import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
from  torch_geometric.nn import conv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import pandas as pd


from collections import Counter
import numpy as np
import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

#from gnn import GNN

import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import random

from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import DataLoader
import os
import os.path as osp


# In[3]:


import torch

def convert_to_coo(mat):
    src = mat

    rowptr, col, value = src.csr()

    row = src.storage._row
    csr2csc = src.storage._csr2csc
    colptr = src.storage._colptr

    index = torch.stack([row,col])
    value = torch.ones(col.size(0))

    s = torch.sparse_coo_tensor(index, value, (src.size(0), src.size(1)))
    return s
def sum_row(mat):
    I = torch.ones(mat.size(1))
    return torch.sparse.mm(mat, I.reshape(-1,1))

def init_messages(mat, prior):

    author_coo = convert_to_coo(mat)

    author_values = []
    for i in range(author_coo.size(0)):
        values = author_coo[i]* prior[i].item()
        author_values.append(values.coalesce().values()) 

    values = torch.cat(author_values, 0).reshape(-1)

    author_coo_message = torch.sparse_coo_tensor(author_coo.coalesce().indices(), values,(author_coo.size(0), author_coo.size(1))) 
    return author_coo_message

def convert_to_mat(df_mats,value=0,axis=0, sizes = [1,1]):
    print('df_mats.shape:{}, axis:{}, value:{}'.format(df_mats.shape, axis, value))
    df_mat = df_mats[df_mats[axis] == value].drop([axis],axis=1)
    dims = list(df_mat.columns.values)
    dims.pop()
    dims = dims[::-1]
    if len(dims) > 1:
        print('mat sizes: {}'.format(sizes))
        print('actual mat sizes: {},{}'.format(df_mat[dims[0]].max(), df_mat[dims[1]].max()))
        
        m = np.zeros(sizes)
        for index,row in df_mat.iterrows():
            m[int(row[dims[0]]),int(row[dims[1]])] = row['prob']
    else:
        df_mat = df_mats[df_mats[axis] == value].drop([axis],axis=1)
        dims = list(df_mat.columns.values)
        dims.pop()
        m = np.zeros(sizes[0])
        for index,row in df_mat.iterrows():
            m[int(row[dims[0]])] = row['prob']
        
    return m

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



dataset = PygNodePropPredDataset(name='ogbn-mag',root='data')

data = dataset[0]

split_idx = dataset.get_idx_split()

# We do not consider those attributes for now.
#data.node_year_dict = None
#data.edge_reltype_dict = None


data.adj_t_dict = {}
paper_fields = None
institution_author = None
author_paper = None
citation_paper = None
for keys, (row, col) in data.edge_index_dict.items():
    sizes = (data.num_nodes_dict[keys[0]], data.num_nodes_dict[keys[2]])
    adj = SparseTensor(row=row, col=col, sparse_sizes=sizes)

    if keys[0] != keys[2]:
        data.adj_t_dict[keys] = adj.t()
        data.adj_t_dict[(keys[2], 'to', keys[0])] = adj
    else:
        data.adj_t_dict[keys] = adj.to_symmetric()
        
data.adj_t_dict[('paper', 'to', 'field_of_study')] = convert_to_coo(data.adj_t_dict[('field_of_study', 'to', 'paper')]) 
data.adj_t_dict[('paper', 'to', 'citation')] = torch.cat((data['edge_index_dict'][('paper', 'cites', 'paper')][1].reshape(-1,1),data['edge_index_dict'][('paper', 'cites', 'paper')][0].reshape(-1,1)),1)
data.adj_t_dict[('paper', 'to', 'citation')] = data.adj_t_dict[('paper', 'to', 'citation')].to_sparse()
data.adj_t_dict[('paper','to','topic')] = data.y_dict['paper'].to_sparse()

data.adj_t_dict[('author', 'affiliated_with', 'institution')] = convert_to_coo(data.adj_t_dict[('author', 'affiliated_with', 'institution')]) 

data.adj_t_dict[('institution', 'to', 'author')] = convert_to_coo(data.adj_t_dict[('institution', 'to', 'author')]) 

data.adj_t_dict[('author', 'writes', 'paper')] = convert_to_coo(data.adj_t_dict[('author', 'writes', 'paper')]) 

data.adj_t_dict[('paper', 'to', 'author')] = convert_to_coo(data.adj_t_dict[('paper', 'to', 'author')]) 

data.adj_t_dict[('paper', 'cites', 'paper')] = convert_to_coo(data.adj_t_dict[('paper', 'cites', 'paper')]) 
data.adj_t_dict[('paper', 'to', 'citation')] = data.adj_t_dict[('paper', 'cites', 'paper')].t()

data.adj_t_dict[('paper', 'has_topic', 'field_of_study')] = convert_to_coo(data.adj_t_dict[('paper', 'has_topic', 'field_of_study')])  

data.adj_t_dict[('field_of_study', 'to', 'paper')] = convert_to_coo(data.adj_t_dict[('field_of_study', 'to', 'paper')])  


# In[5]:


data


# In[ ]:


num_classes = len(set(np.array(data['y_dict']['paper'].reshape(-1))))


# In[ ]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='data/Cora', name='Cora')



class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_channels, 8)
        
        self.conv2 = GCNConv(8, out_channels)
        
        #self.conv1 = GATConv(in_channels, 16, heads=16, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        #self.conv2 = GATConv(16 * 16, out_channels, heads=8, concat=False, dropout=0.6)
        
        
        

    def forward(self, data):
        x, edge_index = data['x_dict']['paper'], data['edge_index_dict'][('paper','cites','paper')]
        
        x = F.elu(self.conv1(x, edge_index))
        x = torch.nn.BatchNorm1d(x.size(1))(x)
        x = torch.nn.ReLU()(x)
        x = torch.nn.Dropout(p = 0.2)(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


        return F.log_softmax(x, dim=1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data =dataset[0]
data = data.to(device)

model = Net(data['x_dict']['paper'].size(1), num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    #loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss = F.nll_loss(out, data['y_dict']['paper'])
    loss.backward()
    optimizer.step()
    
model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred.eq(data['y_dict']['paper']).sum().item())
acc = correct / int(data['y_dict']['paper'].size(0))
print('Accuracy: {:.4f}'.format(acc))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # GAT on Cora dataset

# In[ ]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import os.path as osp

from torch_geometric.datasets import Planetoid
dataset = 'Cora'
path = osp.join('data', dataset)
#dataset = Planetoid(path, dataset, transform=np.T.NormalizeFeatures())
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()

        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, out_channels)
        

    def forward(self, x, edge_index):
        #x = F.dropout(x, p=0.6, training=self.training)
        #x = F.elu(self.conv1(x, edge_index))
        #x = F.dropout(x, p=0.6, training=self.training)
        #x = self.conv2(x, edge_index)
        
        x = F.elu(self.conv1(x, edge_index))
        x = torch.nn.BatchNorm1d(x.size(1))(x)
        x = torch.nn.ReLU()(x)
        x = torch.nn.Dropout(p = 0.2)(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
        accs.append(acc)
    return accs


for epoch in range(1, 10+1):
    train(data)
    train_acc, val_acc, test_acc = test(data)
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from torch_scatter import scatter_mean

import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree

import math

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim),                                        torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(),                                        torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)    
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
    
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out


class GCN(torch.nn.Module):
    def __init__(self, in_channels=100, out_channels=100, emb_dim=100,num_conv_layers=5, drop_ratio=0.5):
        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_dim = emb_dim
        self.num_conv_layers = num_conv_layers
        self.drop_ratio = drop_ratio

        # mlp
        module_list = [
            conv.GCNConv(in_channels=self.in_channels, out_channels=self.emb_dim),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = self.drop_ratio),
            
        ]
        
        for i in range(self.num_conv_layers - 1):
            module_list += [conv.GCNConv(in_channels=self.emb_dim, out_channels=self.emb_dim),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = self.drop_ratio),]
            
        # relu is applied in the last layer to ensure positivity
        module_list += [conv.GCNConv(in_channels=self.emb_dim, out_channels=self.out_channels)]
        
        self.gcn = torch.nn.Sequential( *module_list)
    
    def forward(self, x, edge_index):
        output = self.gcn(x, edge_index)
        if self.training:
            return output 
        else:
            # At inference time, relu is applied to output to ensure positivity
            return torch.clamp(output, min=0, max=50)



# In[ ]:



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


# GCN 
I = data['x_dict']['paper'].size(1)
O = len(set(np.array(data['y_dict']['paper'].reshape(-1))))
emb_dim = 100
num_conv_layers = 5
model = GCN(in_channels=I, out_channels=O,             emb_dim=emb_dim,             num_conv_layers=num_conv_layers,             drop_ratio=0.2)


# # GCN

# In[ ]:


import argparse

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator



class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, embeddings, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(embeddings.x, embeddings.adj_t)[train_idx]
    loss = F.nll_loss(out, embeddings.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, embeddings, split_idx, evaluator):
    model.eval()

    out = model(data.x, embeddings.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc


device=0
log_steps=2
use_sage=True
num_layers=2
hidden_channels=256
dropout=0.5
lr=0.01
epochs=100
runs=10

device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


dataset = PygNodePropPredDataset(name='ogbn-mag',root='data')
rel_data = dataset[0]

# We are only interested in paper <-> paper relations.
data = Data(
    x=rel_data.x_dict['paper'],
    edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
    y=rel_data.y_dict['paper'])

data = T.ToSparseTensor()(data)
data.adj_t = data.adj_t.to_symmetric()

split_idx = dataset.get_idx_split()
train_idx = split_idx['train']['paper'].to(device)

if use_sage:
    model = SAGE(data.num_features, hidden_channels,
                 dataset.num_classes, num_layers,
                 dropout).to(device)
else:
    model = GCN(data.num_features, hidden_channels,
                dataset.num_classes, num_layers,
                dropout).to(device)

    # Pre-compute GCN normalization.
    adj_t = data.adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t

data = data.to(device)

evaluator = Evaluator(name='ogbn-mag')
logger = Logger(runs)

for run in range(runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, 1 + epochs):
        loss = train(model, data, train_idx, optimizer)
        result = test(model, data, split_idx, evaluator)
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





# In[ ]:





# In[ ]:


device=0
num_layers=2
hidden_channels=64
dropout=0.5
lr=0.01
epochs=3
runs=10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


logger = Logger(runs)


# In[ ]:


## MLP

class MLP(torch.nn.Module):
    def __init__(self, num_mlp_layers = 5, emb_dim = 300, drop_ratio = 0):
        super(MLP, self).__init__()
        self.num_mlp_layers = num_mlp_layers
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio 

        # mlp
        module_list = [
            torch.nn.Linear(128, self.emb_dim),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = self.drop_ratio),
        ]

        for i in range(self.num_mlp_layers - 1):
            module_list += [torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = self.drop_ratio)]
        
        # relu is applied in the last layer to ensure positivity
        module_list += [torch.nn.Linear(self.emb_dim, 1)]

        self.mlp = torch.nn.Sequential(
            *module_list
        )
    
    def forward(self, x):
        output = self.mlp(x)
        if self.training:
            return output 
        else:
            # At inference time, relu is applied to output to ensure positivity
            return torch.clamp(output, min=0, max=50)



        
reg_criterion = torch.nn.L1Loss()

def train(model, device, loader, optimizer):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        #x = x.to(device).to(torch.float32)
        #y = y.to(device)

        pred = model(x).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, y)
        loss.backward()
        optimizer.step()

        #loss_accum += loss.detach().cpu().item()
        loss_accum += loss.item()

    return loss_accum / (step + 1)

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        #x = x.to(device).to(torch.float32)
        #y = y.to(device)

        with torch.no_grad():
            pred = model(x).view(-1,)

        #y_true.append(y.view(pred.shape).detach().cpu())
        #y_pred.append(pred.detach().cpu())
        y_true.append(y)
        y_pred.append(pred)

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)
    y_true = y_true.reshape(-1,1)
    y_pred = y_pred.reshape(-1,1)
    print('y_true.ndim={}, y_pred.ndim={}'.format(y_true.ndim, y_pred.ndim))

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)['acc'] # evaluator.eval(input_dict)["mae"]

def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch 
        #x = x.to(device).to(torch.float32)

        with torch.no_grad():
            pred = model(x).view(-1,)

        #y_pred.append(pred.detach().cpu())
        y_pred.append(pred)

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred


shared_params = {
    'num_layers': 5,
    'emb_dim': 300,
    'drop_ratio': 0,
    'graph_pooling': 'sum'
}
#device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")


# In[ ]:





# In[ ]:


root = 'data/ogbn_mag'
X, Y, edge_index = data['x_dict']['paper'], data['y_dict']['paper'], data['edge_index_dict'][('paper','cites','paper')]

train_dataset = TensorDataset(X, Y)

valid_dataset = train_dataset
test_dataset = train_dataset

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers = 1)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers = 1)

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers = 1)
os.makedirs(osp.join(root, f'checkpoint'), exist_ok = True)


# In[ ]:


from torch.utils.data import random_split

from torch.utils.data import SubsetRandomSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
evaluator = Evaluator(name='ogbn-mag')

indices = list(range(X.size(0)))
split = int(np.floor(X.size(0)//3))
train_indices, val_indices, test_indices = indices[:split], indices[split:2*split],indices[2*split:]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(train_dataset,sampler=train_sampler, batch_size=256, shuffle=False, num_workers = 1)
valid_loader = DataLoader(train_dataset,sampler=valid_sampler, batch_size=256, shuffle=False, num_workers = 1)
test_loader = DataLoader(train_dataset,sampler=test_sampler, batch_size=256, shuffle=False, num_workers = 1)
os.makedirs(osp.join(root, f'checkpoint'), exist_ok = True)


# In[ ]:





# In[ ]:



## run MLP

model = MLP(num_mlp_layers=5, emb_dim=300, drop_ratio=0.2).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f'#Params: {num_params}')

optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter(log_dir=osp.join(root, f'log'))

best_valid_mae = 1000

scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

for epoch in range(1, 10 + 1):
    print("=====Epoch {}".format(epoch))
    print('Training...')
    train_mae = train(model, device, train_loader, optimizer)

    print('Evaluating...')
    valid_mae = eval(model, device, valid_loader, evaluator)

    print({'Train': train_mae, 'Validation': valid_mae})


    writer.add_scalar('valid/mae', valid_mae, epoch)
    writer.add_scalar('train/mae', train_mae, epoch)

    if valid_mae < best_valid_mae:
        best_valid_mae = valid_mae
        
        print('Saving checkpoint...')
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae, 'num_params': num_params}
        torch.save(checkpoint, os.path.join(root, 'checkpoint.pt'))


        print('Predicting on test data...')
        y_pred = test(model, device, test_loader)
        print('Saving test submission file...')
        #evaluator.save_test_submission({'y_pred': y_pred}, osp.join(root, f'test'))
        
        torch.save({'y_pred': y_pred}, os.path.join(root,f'test/test.pt'))

    scheduler.step()

    print(f'Best validation MAE so far: {best_valid_mae}')

writer.close()


# # RGCN

# In[ ]:


# sparse data set batching
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import random
from scipy.sparse import csr_matrix

import gc
gc.collect()
torch.cuda.empty_cache()



class SparseData(Dataset):
    def __init__(self, indices, values, dims, y):
        self.rows = indices.size(1)
        # csr format for being able to slice the matrix
        self.x = csr_matrix((values, (indices[0], indices[1])), shape=dims)
        self.y = y

    def __len__(self):
        return self.rows

    def __getitem__(self, idx):
        x_coo = self.x[idx, :].tocoo()
        ind = torch.LongTensor([x_coo.row, x_coo.col])
        values = torch.FloatTensor(x_coo.data)
        mat = torch.sparse.FloatTensor(ind, values, list(x_coo.shape)).to_dense()
        labels = self.y.reshape(-1,1)[idx]
        
        return (mat,labels)
    
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

class FullConnectedModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FullConnectedModule, self).__init__()
        I = in_channels
        O= out_channels
        E = I
        self.conv1_1 = torch.nn.Conv1d(I, E//4, 1)
        self.relu1_1 = torch.nn.ReLU(inplace=True)
        E = E//4
        self.conv1_2 = torch.nn.Conv1d(E, E//4, 1)
        self.relu1_2 = torch.nn.ReLU(inplace=True)
        E = E//4
        self.conv1_3 = torch.nn.Conv1d(E, E//4, 1)
        self.relu1_3 = torch.nn.ReLU(inplace=True)
        E = E//4
        self.conv1_5 = torch.nn.Conv1d(E, E//2, 1)
        self.relu1_5 = torch.nn.ReLU(inplace=True)
        E = E//2
        self.score = torch.nn.Conv1d(E, O, 1)
        
    def forward(self, network):
        h = self.relu1_1(self.conv1_1(network))
        h = self.relu1_2(self.conv1_2(h))
        h = self.relu1_3(self.conv1_3(h))
        h = self.relu1_5(self.conv1_5(h))
        h = self.score(h)
        return h

class CONV1D(FullConnectedModule):
    def __init__(self, in_channels, out_channels):
        super(CONV1D,self).__init__(in_channels, out_channels)
        
    def forward(self, network):
        out = super(CONV1D, self).forward(network)
        return out

indices = data['edge_index_dict'][('paper','has_topic','field_of_study')]
values = torch.ones(indices.size(1))
dims = (data['num_nodes_dict']['paper'],data['num_nodes_dict']['field_of_study'])
y_true = data['y_dict']['paper'].reshape(-1)

train_data = SparseData(indices, values, dims, y_true)
train_loader = DataLoader(train_data, batch_size=8, shuffle=False, num_workers=0)
topic_len = len(set(list(data['y_dict']['paper'].reshape(-1).numpy())))
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CONV1D(data['num_nodes_dict']['field_of_study'],topic_len)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

EPOCHS =2

for e in tqdm(range(1, EPOCHS+1)):
    
    loss_total = 0
    y_pred = []
    for batch_i, batch in enumerate(train_loader):
        torch.cuda.empty_cache()
        batch_x, batch_y = batch
        
        batch_x = batch_x.reshape(-1,data['num_nodes_dict']['field_of_study'],1)
        batch_y = batch_y.reshape(-1)
        #print(f'batch_x={batch_x}, batch_y = {batch_y}')
        optimizer.zero_grad()
        pred = model(batch_x).view(-1).reshape(-1, topic_len)
        loss = torch.nn.functional.cross_entropy(model(batch_x).view(-1).reshape(-1, topic_len), batch_y)
        loss.backward()
        loss_total += loss.item()
        optimizer.step()
        y_pred.append(pred.argmax(dim=-1))
        print(f'loss = {loss.item()}')
        
    print(f'loss = {loss_total}')
    y_pred = torch.cat(y_pred)
    acc = y_pred == y_true
    print(f'accuracy = {acc.sum()/len(acc)}')
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




