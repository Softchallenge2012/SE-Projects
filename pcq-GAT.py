#!/usr/bin/env python
# coding: utf-8

# In[39]:


import argparse

import torch
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import matplotlib.pyplot as plt


# In[40]:


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


# In[41]:


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
import os.path as osp


# In[42]:


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



# In[43]:


from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import DataLoader


# In[44]:


pyg_dataset = PCQM4MDataset(root='data/', only_smiles=True)

fp_processed_dir = osp.join(pyg_dataset.folder, 'fp_processed')
fp_processed_file = osp.join(fp_processed_dir, f'data_radius2.pt')
print(fp_processed_file)

alg_name = ''

steps_train = []
steps_test = []
loss_train_MLP = []
loss_test_MLP = []
loss_train_GCN = []
loss_test_GCN = []
loss_train_GIN = []
loss_test_GIN = []
loss_train_GCNV = []
loss_test_GCNV = []
loss_train_GINV = []
loss_test_GINV = []


# In[7]:


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
    torch.save(data_dict, fp_processed_file)


# In[8]:


"""
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

"""

"""
>>>>output
tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 1, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 1, 0,  ..., 0, 0, 0]], dtype=torch.int8)
tensor([3.0477, 4.4110, 4.6395,  ...,    nan,    nan,    nan],
       dtype=torch.float64)
torch.Size([3803453, 2048])
torch.Size([3803453])

"""


# In[9]:



fp_processed_dir = osp.join('data/pcqm4m_kddcup2021', 'fp_processed')
fp_processed_file = osp.join(fp_processed_dir, f'data_radius2.pt')
print(fp_processed_file)


# In[10]:


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

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim),                                        torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
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

        #self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.linear = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim),                                        torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
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


# linear model
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


# In[11]:


data_dict = torch.load(fp_processed_file)
X, Y = data_dict['X'], data_dict['Y']

split_idx = pyg_dataset.get_idx_split()
### automatic evaluator. takes dataset name as input
evaluator = PCQM4MEvaluator()

train_dataset = TensorDataset(X[split_idx['train']], Y[split_idx['train']])

valid_dataset = TensorDataset(X[split_idx['valid']], Y[split_idx['valid']])
test_dataset = TensorDataset(X[split_idx['test']], Y[split_idx['test']])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers = 0)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers = 0)

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers = 0)
os.makedirs(osp.join(fp_processed_dir, f'checkpoint'), exist_ok = True)


# In[12]:


X.shape, Y.shape


# In[13]:


### automatic evaluator. takes dataset name as input
evaluator = PCQM4MEvaluator()


# In[14]:


## MLP
    
reg_criterion = torch.nn.L1Loss()

def train(model, device, loader, optimizer):
    model.train()
    loss_accum = 0
    
    global steps_train
    global loss_train_MLP
    steps_train = []
    loss_train_MLP = []
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        x = x.to(device).to(torch.float32)
        y = y.to(device)

        pred = model(x).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, y)
        if step > 100 and  step % 100 == 0:
            steps_train.append(step)
            if alg_name == 'MLP':
                loss_train_MLP.append(loss.detach().cpu().item())
            
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()


    ETAS = ['Train']

    plt.plot(np.array(steps_train).T, np.array(loss_train_MLP).T, '.-')

    plt.plot(np.linspace(-0.4, 2.0), np.linspace(-0.4, 2.0), 'k:')
    plt.legend(ETAS, title=r'Legend')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.show()
    return loss_accum / (step + 1)

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    
    global steps_test
    global loss_test_MLP
    steps_test = []
    loss_test_MLP = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        x = x.to(device).to(torch.float32)
        y = y.to(device)

        with torch.no_grad():
            pred = model(x).view(-1,)
                
        loss = reg_criterion(pred, y)
        if step > 100 and step % 100 == 0:
            steps_test.append(step)
            if alg_name == 'MLP':
                loss_test_MLP.append(loss.detach().cpu().item())


        y_true.append(y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    ETAS = ['Test']

    plt.plot(np.array(steps_test).T, np.array(loss_test_MLP).T, '.-')
    plt.plot(np.linspace(-0.4, 2.0), np.linspace(-0.4, 2.0), 'k:')
    plt.legend(ETAS, title=r'Legend')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.show()

    return evaluator.eval(input_dict)["mae"]

def test(model, device, loader):
    model.eval()
    y_pred = []

    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch 
        x = x.to(device).to(torch.float32)
        y = y.to(device)

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

## run MLP

num_params = sum(p.numel() for p in model.parameters())
print(f'#Params: {num_params}')

optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter(log_dir=osp.join(fp_processed_dir, f'log'))

best_valid_mae = 1000

scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

alg_name = 'MLP'
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




# In[15]:


###GNN

reg_criterion = torch.nn.L1Loss()

def train(model, device, loader, optimizer):
    model.train()
    loss_accum = 0
    
    global steps_train
    global loss_train_GIN
    global loss_train_GCN
    global loss_train_GINV
    global loss_train_GCNV
    
    steps_train = []
    if alg_name == 'GIN':
        loss_train_GIN = []
    if alg_name == 'GCN':
        loss_train_GCN = []
    if alg_name == 'GINV':
        loss_train_GINV = []
    if alg_name == 'GCNV':
        loss_train_GCNV = []
    

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        if step > 100 and step % 100 == 0:
            steps_train.append(step)
            if alg_name == 'GIN':
                loss_train_GIN.append(loss.detach().cpu().item())
            if alg_name == 'GCN':
                loss_train_GCN.append(loss.detach().cpu().item())
            if alg_name == 'GINV':
                loss_train_GINV.append(loss.detach().cpu().item())
            if alg_name == 'GCNV':
                loss_train_GCNV.append(loss.detach().cpu().item())


        loss_accum += loss.detach().cpu().item()

    ETAS = ['Train']
    
    if alg_name == 'GIN':
        plt.plot(np.array(steps_train).T, np.array(loss_train_GIN).T, '.-')
    if alg_name == 'GCN':
        plt.plot(np.array(steps_train).T, np.array(loss_train_GCN).T, '.-')
    if alg_name == 'GINV':
        plt.plot(np.array(steps_train).T, np.array(loss_train_GINV).T, '.-')
    if alg_name == 'GCNV':
        plt.plot(np.array(steps_train).T, np.array(loss_train_GCNV).T, '.-')

    
    
    plt.plot(np.linspace(-0.4, 2.0), np.linspace(-0.4, 2.0), 'k:')
    plt.legend(ETAS, title=r'Legend')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.show()
    return loss_accum / (step + 1)

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    
    global steps_test
    global loss_test_GIN
    global loss_test_GCN
    global loss_test_GINV
    global loss_test_GCNV
    
    steps_test = []
    if alg_name == 'GIN':
        loss_test_GIN = []
    if alg_name == 'GCN':
        loss_test_GCN = []
    if alg_name == 'GINV':
        loss_test_GINV = []
    if alg_name == 'GCNV':
        loss_test_GCNV = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)
            
        
        if step > 100 and step % 100 == 0:
            loss = reg_criterion(pred, batch.y)
            steps_test.append(step)
            if alg_name == 'GIN':
                loss_test_GIN.append(loss.detach().cpu().item())
            if alg_name == 'GCN':
                loss_test_GCN.append(loss.detach().cpu().item())
            if alg_name == 'GINV':
                loss_test_GINV.append(loss.detach().cpu().item())
            if alg_name == 'GCNV':
                loss_test_GCNV.append(loss.detach().cpu().item())

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    
    ETAS = ['Evaluate']
    if alg_name == 'GIN':
        plt.plot(np.array(steps_test).T, np.array(loss_test_GIN).T, '.-')
    if alg_name == 'GCN':
        plt.plot(np.array(steps_test).T, np.array(loss_test_GCN).T, '.-')
    if alg_name == 'GINV':
        plt.plot(np.array(steps_test).T, np.array(loss_test_GINV).T, '.-')
    if alg_name == 'GCNV':
        plt.plot(np.array(steps_test).T, np.array(loss_test_GCNV).T, '.-')


    
    
    plt.plot(np.linspace(-0.4, 2.0), np.linspace(-0.4, 2.0), 'k:')
    plt.legend(ETAS, title=r'Legend')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.show()
    
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


# In[16]:


dataset = PygPCQM4MDataset(root = 'data/')
root = osp.join(dataset.folder, 'processed')
print(root)


split_idx = dataset.get_idx_split()

### automatic evaluator. takes dataset name as input
evaluator = PCQM4MEvaluator()
device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")


subset_ratio = 0.1
subset_idx = torch.randperm(len(split_idx["train"]))[:int(subset_ratio*len(split_idx["train"]))]
train_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=256, shuffle=True, num_workers = 0)

valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=256, shuffle=False, num_workers = 0)

test_loader = DataLoader(dataset[split_idx["test"]], batch_size=256, shuffle=False, num_workers = 0)

os.makedirs(osp.join(root,'checkpoint'), exist_ok = True)


# In[ ]:





# In[17]:


# GIN 

model = GNN(gnn_type = 'gin', virtual_node = False, **shared_params).to(device)


num_params = sum(p.numel() for p in model.parameters())
print(f'#Params: {num_params}')

optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter(log_dir=osp.join(root,'log'))

best_valid_mae = 1000

scheduler = StepLR(optimizer, step_size=30, gamma=0.25)
epochs = 10

alg_name = 'GIN'
for epoch in range(1, epochs + 1):
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
        torch.save(checkpoint, os.path.join(osp.join(root,'checkpoint'), 'checkpoint.pt'))

        
        print('Predicting on test data...')
        y_pred = test(model, device, test_loader)
        print('Saving test submission file...')
        evaluator.save_test_submission({'y_pred': y_pred}, osp.join(root,'test'))
 

    scheduler.step()

    print(f'Best validation MAE so far: {best_valid_mae}')


writer.close()


# In[18]:


# GCN 


model = GNN(gnn_type = 'gcn', virtual_node = False, **shared_params).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f'#Params: {num_params}')

optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter(log_dir=osp.join(root,'log'))

best_valid_mae = 1000

scheduler = StepLR(optimizer, step_size=30, gamma=0.25)
epochs = 10

alg_name = 'GCN'
for epoch in range(1, epochs + 1):
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
        torch.save(checkpoint, os.path.join(osp.join(root,'checkpoint'), 'checkpoint.pt'))

        
        print('Predicting on test data...')
        y_pred = test(model, device, test_loader)
        print('Saving test submission file...')
        evaluator.save_test_submission({'y_pred': y_pred}, osp.join(root,'test'))
 

    scheduler.step()

    print(f'Best validation MAE so far: {best_valid_mae}')


writer.close()


# In[19]:


#GIN Virtual

model = GNN(gnn_type = 'gin', virtual_node = True, **shared_params).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f'#Params: {num_params}')

optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter(log_dir=osp.join(root,'log'))

best_valid_mae = 1000

scheduler = StepLR(optimizer, step_size=30, gamma=0.25)
epochs = 10

alg_name = 'GINV'
for epoch in range(1, epochs + 1):
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
        torch.save(checkpoint, os.path.join(osp.join(root,'checkpoint'), 'checkpoint.pt'))

        
        print('Predicting on test data...')
        y_pred = test(model, device, test_loader)
        print('Saving test submission file...')
        evaluator.save_test_submission({'y_pred': y_pred}, osp.join(root,'test'))
 

    scheduler.step()

    print(f'Best validation MAE so far: {best_valid_mae}')


writer.close()


# In[20]:


# GCN Virtual


model = GNN(gnn_type = 'gcn', virtual_node = True, **shared_params).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f'#Params: {num_params}')

optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter(log_dir=osp.join(root,'log'))

best_valid_mae = 1000

scheduler = StepLR(optimizer, step_size=30, gamma=0.25)
epochs = 10

alg_name = 'GCNV'
for epoch in range(1, epochs + 1):
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
        torch.save(checkpoint, os.path.join(osp.join(root,'checkpoint'), 'checkpoint.pt'))

        
        print('Predicting on test data...')
        y_pred = test(model, device, test_loader)
        print('Saving test submission file...')
        evaluator.save_test_submission({'y_pred': y_pred}, osp.join(root,'test'))
 

    scheduler.step()

    print(f'Best validation MAE so far: {best_valid_mae}')


writer.close()


# In[31]:


ETAS = ['GIN','GCN','GINV','GCNV']

plt.plot(np.array(steps_train).T, np.array([loss_train_GIN,loss_train_GCN,loss_train_GINV,loss_train_GCNV]).T, '.-')

plt.plot(np.linspace(-0.4, 2.0), np.linspace(-0.4, 2.0), 'k:')
plt.legend(ETAS, title=r'Legend')
plt.xlabel('steps')
plt.ylabel('loss')
plt.show()


# In[32]:


ETAS = ['GIN','GCN','GINV','GCNV']

plt.plot(np.array(steps_test).T, np.array([loss_test_GIN,loss_test_GCN,loss_test_GINV,loss_test_GCNV]).T, '.-')

plt.plot(np.linspace(-0.4, 2.0), np.linspace(-0.4, 2.0), 'k:')
plt.legend(ETAS, title=r'Legend')
plt.xlabel('steps')
plt.ylabel('loss')
plt.show()


# In[37]:


dataset.data


# ## GAT

# In[34]:


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


# In[38]:



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
cuda=False
fastmode=False
sparse=False
seed=72
epochs=10
lr=0.005
weight_decay=5e-4
hidden=8
nb_heads=8
dropout=0.6
alpha=0.2
patience=100


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)


# Model and optimizer
model = GAT(nfeat=features.shape[1], 
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
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

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




