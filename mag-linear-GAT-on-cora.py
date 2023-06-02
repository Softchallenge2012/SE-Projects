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


# In[ ]:





# In[ ]:





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


# In[3]:


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
class GCN(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GCN, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim),                                        torch.nn.BatchNorm1d(emb_dim),                                        torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)    
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
    
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out


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


# In[4]:


device=0
num_layers=2
hidden_channels=64
dropout=0.5
lr=0.01
epochs=3
runs=10


# In[5]:


logger = Logger(runs)


# In[6]:



dataset = PygNodePropPredDataset(name='ogbn-mag',root='data')

data = dataset[0]


# In[7]:


data


# In[8]:


root = 'data/ogbn_mag'
X, Y = data['x_dict']['paper'], data.y_dict['paper']


train_dataset = TensorDataset(X, Y)

valid_dataset = train_dataset
test_dataset = train_dataset

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers = 1)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers = 1)

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers = 1)
os.makedirs(osp.join(root, f'checkpoint'), exist_ok = True)


# In[9]:


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


# In[10]:


## GCN

reg_criterion = torch.nn.L1Loss()

def train(model, device, loader, adj, optimizer):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        x = x.to(device).to(torch.float32)
        y = y.to(device)
        adj= adj[('paper','cite_by','paper')].to(device)

        pred = model(x, adj).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        x = x.to(device).to(torch.float32)
        y = y.to(device)

        with torch.no_grad():
            pred = model(x).view(-1,)

        y_true.append(y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

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
        x = x.to(device).to(torch.float32)

        with torch.no_grad():
            pred = model(x).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred


shared_params = {
    'num_layers': 5,
    'emb_dim': 300,
    'drop_ratio': 0,
    'graph_pooling': 'sum'
}
device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

model = GCN(num_mlp_layers=5, emb_dim=300, drop_ratio=0).to(device)


# In[ ]:



## run GCN

num_params = sum(p.numel() for p in model.parameters())
print(f'#Params: {num_params}')

optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter(log_dir=osp.join(root, f'log'))

best_valid_mae = 1000

scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

for epoch in range(1, 10 + 1):
    print("=====Epoch {}".format(epoch))
    print('Training...')
    train_mae = train(model, device, train_loader, data['edge_index_dict'], optimizer)

    print('Evaluating...')
    valid_mae = eval(model, device, valid_loader, evaluator)

    print({'Train': train_mae, 'Validation': valid_mae})


    writer.add_scalar('valid/mae', valid_mae, epoch)
    writer.add_scalar('train/mae', train_mae, epoch)

    if valid_mae < best_valid_mae:
        best_valid_mae = valid_mae
        
        print('Saving checkpoint...')
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),                       'optimizer_state_dict': optimizer.state_dict(),                       'scheduler_state_dict': scheduler.state_dict(),                       'best_val_mae': best_valid_mae, 'num_params': num_params}
        torch.save(checkpoint, os.path.join(root, 'checkpoint.pt'))


        print('Predicting on test data...')
        y_pred = test(model, device, test_loader)
        print('Saving test submission file...')
        #evaluator.save_test_submission({'y_pred': y_pred}, osp.join(root, f'test'))
        
        torch.save({'y_pred': y_pred}, os.path.join(root,f'test/test.pt'))

    scheduler.step()

    print(f'Best validation MAE so far: {best_valid_mae}')

writer.close()


# In[ ]:


## MLP

reg_criterion = torch.nn.L1Loss()

def train(model, device, loader, optimizer):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        x = x.to(device).to(torch.float32)
        y = y.to(device)

        pred = model(x).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        x = x.to(device).to(torch.float32)
        y = y.to(device)

        with torch.no_grad():
            pred = model(x).view(-1,)

        y_true.append(y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

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
        x = x.to(device).to(torch.float32)

        with torch.no_grad():
            pred = model(x).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred


shared_params = {
    'num_layers': 5,
    'emb_dim': 300,
    'drop_ratio': 0,
    'graph_pooling': 'sum'
}
device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

model = MLP(num_mlp_layers=5, emb_dim=300, drop_ratio=0).to(device)


# In[ ]:





# In[ ]:





# In[ ]:



## run MLP

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
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),                       'optimizer_state_dict': optimizer.state_dict(),                       'scheduler_state_dict': scheduler.state_dict(),                       'best_val_mae': best_valid_mae, 'num_params': num_params}
        torch.save(checkpoint, os.path.join(root, 'checkpoint.pt'))


        print('Predicting on test data...')
        y_pred = test(model, device, test_loader)
        print('Saving test submission file...')
        #evaluator.save_test_submission({'y_pred': y_pred}, osp.join(root, f'test'))
        
        torch.save({'y_pred': y_pred}, os.path.join(root,f'test/test.pt'))

    scheduler.step()

    print(f'Best validation MAE so far: {best_valid_mae}')

writer.close()


# In[ ]:


data


# ## GAT

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))

    
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


# In[ ]:



from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import scipy.sparse as sp




# Training settings
#no-cuda=False
cuda=True
fastmode=False
sparse=False
seed=72
epochs=10
lr=0.005
weight_decay=5e-4
hidden=8
nb_heads=8
dropout=0
alpha=0.2
patience=100


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# Load data
dataset = 'data/cora/'
adj, features, labels, idx_train, idx_val, idx_test = load_data()

"""
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
        
adj = convert_to_coo(data.adj_t_dict[('paper', 'cites', 'paper')]) 

features = data['x_dict']['paper']
labels = data['y_dict']['paper'].reshape(-1)

indices = list(range(labels.size(0)))
split = int(np.floor(labels.size(0)//3))
idx_train, idx_test, idx_val = indices[:split], indices[split:2*split],indices[2*split:]




"""


# Model and optimizer
"""
    model = SpGAT(nfeat=features.shape[1], 
                nhid=hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=dropout, 
                nheads=nb_heads, 
                alpha=alpha)
                
   model = GAT(nfeat=features.shape[1], 
                nhid=hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=dropout, 
                nheads=nb_heads, 
                alpha=alpha)             

"""
model = SpGAT(nfeat=features.shape[1], 
            nhid=hidden, 
            nclass=int(labels.max()) + 1, 
            dropout=dropout, 
            nheads=nb_heads, 
            alpha=alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=lr, 
                       weight_decay=weight_decay)

if cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()


features, adj, labels = Variable(features), Variable(adj), Variable(labels)


# In[ ]:



# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = epochs + 1
best_epoch = 0
for epoch in range(epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))


# Testing
compute_test()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




