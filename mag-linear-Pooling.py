#!/usr/bin/env python
# coding: utf-8

# In[2]:


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





# In[3]:


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


# In[5]:


device=0
num_layers=2
hidden_channels=64
dropout=0.5
lr=0.01
epochs=3
runs=10


# In[6]:


logger = Logger(runs)


# In[21]:



dataset = PygNodePropPredDataset(name='ogbn-mag',root='data')

data = dataset[0]


# In[22]:


data


# In[9]:


root = 'data/ogbn_mag'
X, Y = data['x_dict']['paper'], data.y_dict['paper']


train_dataset = TensorDataset(X, Y)

valid_dataset = train_dataset
test_dataset = train_dataset

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers = 1)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers = 1)

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers = 1)
os.makedirs(osp.join(root, f'checkpoint'), exist_ok = True)


# In[10]:


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


# In[7]:


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


# In[8]:





# In[12]:





# In[13]:



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


# # GCN

# In[24]:




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


def train(model, train_loader,x, y, optimizer):
    """
    
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

    """
    model.train()

    pbar = tqdm(total=x.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')
    
    x = x.to(device)
    y = y.to(device)

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        
        for i, (edge_index, e_id, size) in enumerate(adjs):
        
            edge_index = edge_index.to(device)
            print(edge_index)

            #adjs = [adj.to(device) for adj in adjs]
            optimizer.zero_grad()
            out = model(x, edge_index)
            #y = y_global[n_id][:batch_size].squeeze()
            loss = F.nll_loss(out[n_id], y[n_id])
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().item() * batch_size
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / paper_train_idx.size(0)

    return loss




@torch.no_grad()
def test(model, dataloader, x, y, evaluator):
    model.eval()
    
    pbar = tqdm(total=x.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')
    
    x = x.to(device)
    y = y.to(device)

    total_loss = 0
    y_pred = []
    for batch_size, n_id, adjs in data_loader:
        n_id = n_id.to(device)
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(x, adjs)
        y_pred.append(out.argmax(dim=-1, keepdim=True).detach().cpu()) 
        optimizer.step()
        pbar.update(batch_size)
        
    y_pred = torch.cat(y_pred).reshape(-1)

    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc


device=0
log_steps=1
use_sage=False
num_layers=2
hidden_channels=256
dropout=0.5
lr=0.01
epochs=100
runs=10

device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

dataset = PygNodePropPredDataset(name='ogbn-mag', root='data')
rel_data = dataset[0]

# We are only interested in paper <-> paper relations.
data = Data(
    x=rel_data.x_dict['paper'].to(torch.float),
    edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')].to(torch.long),
    y=rel_data.y_dict['paper'].to(torch.long))

#data = T.ToSparseTensor()(data)
#data.adj_t = data.adj_t.to_symmetric()

split_idx = dataset.get_idx_split()
train_idx = split_idx['train']['paper'].to(device)

test_idx = split_idx['test']['paper'].to(device)

if use_sage:
    model = SAGE(data.num_features, hidden_channels,
                 dataset.num_classes, num_layers,
                 dropout).to(device)
else:
    model = GCN(data.num_features, hidden_channels,
                dataset.num_classes, num_layers,
                dropout).to(device)

    """
    # Pre-compute GCN normalization.
    adj_t = data.adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t
    """


#data = data.to(device)
train_loader = NeighborSampler(data.edge_index, node_idx=torch.Tensor(range(data.x.size(0))).to(torch.long),
                               sizes=[25, 20], batch_size=128, shuffle=True,
                               num_workers=0)

test_loader = NeighborSampler(data.edge_index, node_idx=torch.Tensor(range(data.x.size(0))).to(torch.long),
                               sizes=[25, 20], batch_size=128, shuffle=True,
                               num_workers=0)

evaluator = Evaluator(name='ogbn-mag')
logger = Logger(runs)

import gc
gc.collect()
torch.cuda.empty_cache()


for run in range(runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, 1 + epochs):
        torch.cuda.empty_cache()

        loss = train(model, train_loader,data.x, data.y, optimizer)
        result = test(model, test_loader,data.x, data.y, evaluator)
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


# In[25]:


type(rel_data.edge_index_dict[('paper', 'cites', 'paper')])


# In[ ]:





# In[ ]:





# In[ ]:




