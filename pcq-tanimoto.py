#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse

import torch
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import logger


# In[1]:


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


### importing OGB-LSC
from ogb.lsc import PygPCQM4MDataset,PCQM4MDataset, PCQM4MEvaluator
from ogb.utils import smiles2graph


# In[7]:


from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import DataLoader


# In[2]:


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina


# In[9]:


pyg_dataset = PCQM4MDataset(root='data', only_smiles=True)


# In[10]:


import os.path as osp

fp_processed_dir = osp.join(pyg_dataset.folder, 'fp_processed')
fp_processed_file = osp.join(fp_processed_dir, f'data_radius2.pt')
print(fp_processed_file)


# In[6]:


dataset = pyg_dataset
if not osp.exists(fp_processed_file):
    ### automatic dataloading and splitting
    os.makedirs(fp_processed_dir, exist_ok=True)

    x_list = []
    y_list = []
    for i in tqdm(range(len(dataset))):
        smiles, y = dataset[i]
        mol = Chem.MolFromSmiles(smiles)
        x = torch.tensor(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2)), dtype=torch.int8)
        y_list.append(y)
        x_list.append(x)

    X = torch.stack(x_list)
    Y = torch.tensor(y_list)
    print(X)
    print(Y)
    print(X.shape)
    print(Y.shape)
    data_dict = {'X': X, 'Y': Y}
    #torch.save(data_dict, fp_processed_file)


# In[8]:


data_dict = torch.load(fp_processed_file)
X, Y = data_dict['X'], data_dict['Y']

split_idx = pyg_dataset.get_idx_split()
### automatic evaluator. takes dataset name as input
evaluator = PCQM4MEvaluator()

train_dataset = TensorDataset(X[split_idx['train']], Y[split_idx['train']])

valid_dataset = TensorDataset(X[split_idx['valid']], Y[split_idx['valid']])
test_dataset = TensorDataset(X[split_idx['test']], Y[split_idx['test']])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers = 1)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers = 1)

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers = 1)
os.makedirs(osp.join(fp_processed_dir, f'checkpoint'), exist_ok = True)


# In[11]:


import numpy as np

data = X.numpy()
data.shape


# In[ ]:





# # Tanimono Similarity

# In[12]:


from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

#Define clustering setup
def ClusterFps(fps,cutoff=0.2):

    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1,nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        dists.extend([1-x for x in sims])

    # now cluster the data:
    cs = Butina.ClusterData(dists,nfps,cutoff,isDistData=True)
    return cs


# In[13]:


dataset = pyg_dataset

fps = []
for i in tqdm(range(len(dataset))):
    smiles, y = dataset[i]
    mol = Chem.MolFromSmiles(smiles)
    x = AllChem.GetMorganFingerprintAsBitVect(mol, 2,1024)
    fps.append(x)



# In[18]:


clusters=ClusterFps(fps,cutoff=0.4)


# In[ ]:


len(fps)


# In[30]:





# In[45]:


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

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)    
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
    
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layers, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layers (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))
                
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers):
                node_representation += h_list[layer]

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layers, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(),                                                     torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))


    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layers - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers):
                node_representation += h_list[layer]
        
        return node_representation



class GNN(torch.nn.Module):

    def __init__(self, num_tasks = 1, num_layers = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0, JK = "last", graph_pooling = "sum"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''
        super(GNN, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layers, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layers, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)
        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output
        else:
            # At inference time, relu is applied to output to ensure positivity
            return torch.clamp(output, min=0, max=50)




class MLP(torch.nn.Module):
    def __init__(self, num_mlp_layers = 5, emb_dim = 300, drop_ratio = 0):
        super(MLP, self).__init__()
        self.num_mlp_layers = num_mlp_layers
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio 

        # mlp
        module_list = [
            torch.nn.Linear(2048, self.emb_dim),
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


# In[47]:


###GNN

reg_criterion = torch.nn.L1Loss()

def train(model, device, loader, optimizer):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]

def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred


shared_params = {
    'num_layers': 5,
    'emb_dim': 600,
    'drop_ratio': 0,
    'graph_pooling': 'sum'
}
device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

#model = MLP(num_mlp_layers=5, emb_dim=600, drop_ratio=0).to(device)



model = GNN(gnn_type = 'gin', virtual_node = False, **shared_params).to(device)


# In[48]:


import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import random



# In[49]:


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

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]

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
    'emb_dim': 600,
    'drop_ratio': 0,
    'graph_pooling': 'sum'
}
device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

model = MLP(num_mlp_layers=5, emb_dim=600, drop_ratio=0).to(device)



# In[ ]:



num_params = sum(p.numel() for p in model.parameters())
print(f'#Params: {num_params}')

optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter(log_dir=osp.join(fp_processed_dir, f'log'))

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
        torch.save(checkpoint, os.path.join(fp_processed_dir, 'checkpoint.pt'))


        print('Predicting on test data...')
        y_pred = test(model, device, test_loader)
        print('Saving test submission file...')
        evaluator.save_test_submission({'y_pred': y_pred}, osp.join(fp_processed_dir, f'test'))

    scheduler.step()

    print(f'Best validation MAE so far: {best_valid_mae}')

writer.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




