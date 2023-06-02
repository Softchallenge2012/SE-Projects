#!/usr/bin/env python
# coding: utf-8

# In[1]:



import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter, ModuleDict
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.data import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
#from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import copy
import argparse


from collections import Counter
import numpy as np
import pandas as pd
import os
import os.path as osp


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


# In[537]:


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
"""
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
"""

def convert_to_mat(df_mats, value=0, axis=0, sizes=[1,1]):
    print('df_mats.shape:{}, axis:{}, value:{}'.format(df_mats.shape, axis, value))
    df_mat = df_mats[df_mats[axis] == value].drop([axis],axis=1)
    dims = list(df_mat.columns.values)
    dims.pop()
    dims = dims[::-1]
    
    
    df_mat = df_mat[df_mat['prob']>0]
    if len(dims) > 1:
        print('mat sizes: {}'.format(sizes))
        print('actual mat sizes: {},{}'.format(df_mat[dims[0]].max(), df_mat[dims[1]].max()))
        
        row = torch.Tensor(np.array(df_mat[dims[0]]))
        col = torch.Tensor(np.array(df_mat[dims[1]]))

        index = torch.stack([row,col])
        value = torch.Tensor(np.array(df_mat['prob']))

        m = torch.sparse_coo_tensor(index, value, (sizes))
    else:       
        m = np.zeros(sizes[0])
        for index,row in df_mat.iterrows():
            m[int(row[dims[0]])] = row['prob']
            
        m = torch.Tensor(m).reshape(-1,1).to_sparse()
        
    return m
    


# In[ ]:





# In[ ]:





# In[527]:


import numpy as np
from functools import reduce

import numpy as np

def dot_T(x, y):
    return np.dot(x.transpose(), y)

class Node:
    def __init__(self, name):
        self.name = name
        self.cardinality = None
        self.likelihood = None
        self.priors = None
        self.belief = None
        self.parents = []
        self.children = []
        self.m = None

    def add_parent(self, node):
        self.parents.append(node)
        node.children.append(self)

    def __str__(self):
        return self.name

    def message_to_parent(self, parent):

        likelihood = self.get_likelihood()
        parents_priors = np.array([p.message_to_child(self) for p in self.parents if p != parent])
        parent_i = self.parents.index(parent)
        
        #stack = np.vstack([np.dot(convert_to_mat(self.m, value=r, axis=parent_i), parents_priors.prod(axis=0)) for r in range(parent.cardinality)])

        products = []
        for r in range(parent.cardinality):
            if len(self.parents) < 2:
                sizes = [self.cardinality]
            else:
                current_p = [p for p in self.parents if p != parent]
                sizes = [current_p[0].cardinality,self.cardinality]
            
            m = convert_to_mat(self.m, value=r, axis=parent_i, sizes=sizes)
            priors = parents_priors.prod(axis=0)
            product = np.dot(m.to_dense().numpy().transpose(), priors)
            products.append(product)
        stack = np.vstack(products)                  

        
        return np.dot(stack, likelihood)

    def message_to_child(self, child):
        children_messages = np.array([c.message_to_parent(self) for c in self.children if c != child])
        if len(children_messages) > 0:
            unnormalized = (children_messages * self.get_priors()).prod(axis=0)
            #print('{} child {} message: message={}, priors={}, unnormalized={}'.format(self.name,child.name, children_messages, self.get_priors(), unnormalized))
            message = unnormalized/unnormalized.sum()
            return message
        return self.get_priors()

    def get_likelihood(self):
        if self.likelihood is not None:
            return self.likelihood

        incoming_children_messages = np.array([c.message_to_parent(self) for c in self.children])
        return incoming_children_messages.prod(axis=0)

    def get_priors(self):
        if self.priors is not None:
            return self.priors

        parents_messages = [p.message_to_child(self) for p in self.parents]
        priors = reduce(np.dot, [self.m.transpose()]+parents_messages)
        return priors

    def get_belief(self):
        if self.belief is not None:
            return self.belief

        unnormalized = self.get_likelihood() * self.get_priors()
        #print('name={} likelihood={}, priors={}'.format(self.name,self.get_likelihood(), self.get_priors()))
        return unnormalized/unnormalized.sum()


class BeliefPropagation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, layout, edge_types):
        super(BeliefPropagation, self).__init__()
        
        self.layout = layout
        self.edge_types = edge_types
        self.out_channels = out_channels
        self.in_channels = in_channels
        """
        use layout and adj_t_dict to
        form belief network
        institution.prior()
        citation.prior()
        messages: institution to author, author to paper, paper to fields, citation to paper        
        """
        
        m_institution_to_author = edge_types[('institution', 'to', 'author')]
        m_author_to_paper = edge_types[('author', 'writes', 'paper')]
        m_citation_to_paper = edge_types[('paper', 'cites', 'paper')]
        m_paper_to_field = edge_types[('paper', 'to', 'field_of_study')]
        m_paper_to_topic = edge_types[('paper', 'to', 'topic')]
        
        institute = Node("institute")
        institute.cardinality = m_institution_to_author.size(1)
        inst_prior = sum_row(m_institution_to_author.t())
        inst_prior = inst_prior/torch.sum(inst_prior).item()
        institute.priors = inst_prior.reshape(-1).numpy()


        
        topic = Node('topic')
        topic.cardinality = len(set(list(m_paper_to_topic.coalesce().values().numpy())))
        y = m_paper_to_topic.coalesce().values()
        from collections import Counter
        y_counted = Counter(y.numpy())
        y_counted_df = pd.DataFrame.from_dict(y_counted.items())
        y_counted_df = y_counted_df.sort_values(by=[0])
        y_counted_df[1] = y_counted_df[1].apply(lambda s: s*1.0/y.shape[0])
        topic.priors = np.array(y_counted_df[1])
        
        
        author = Node('author')
        author.cardinality = m_institution_to_author.size(0)
        author.m = edge_types['topic_to_author_message']
        author.likelihood = torch.ones(author.cardinality).numpy()
        
        
        citation = Node("citation")
        citation.cardinality = m_citation_to_paper.size(0)
        citation.m = edge_types['topic_to_citation_message']
        citation.likelihood = torch.ones(citation.cardinality).numpy()
        

        fields = Node('fields')
        fields.cardinality = m_paper_to_field.size(1)
        fields.m = edge_types['topic_to_field_message']
        fields.likelihood = torch.ones(fields.cardinality).numpy()
        
        """
        paper = Node('paper')
        paper.cardinality = m_citation_to_paper.size(0)
        paper_prior = sum_row(m_citation_to_paper)
        paper_prior = paper_prior.pow(-1)
        paper_prior[paper_prior.isinf()] = 0
        paper.m = init_messages(m_citation_to_paper, paper_prior)
        
        fields = Node('fields')
        fields.cardinality = m_paper_to_field.size(1)
        fields_prior = sum_row(m_paper_to_field)
        fields_prior = fields_prior.pow(-1)
        fields_prior[fields_prior.isinf()] = 0
        fields.m = init_messages(m_paper_to_field, fields_prior)
        fields.likelihood = tensor.ones(fields.cardinality)
        
        """

        
        self.institute = institute
        self.topic = topic
        self.citation = citation
        self.author = author
        self.fields = fields        
        
        self.author.add_parent(self.institute)
        self.author.add_parent(self.topic)
        self.citation.add_parent(self.topic)
        self.fields.add_parent(self.topic)
        
        nodelist = [self.institute, self.citation, self.topic, self.author, self.fields]
        '''
        for relation in layout:
            parent_name = relation['parent']
            children_name = relation['children']
            
            parent = None
            children = None
            for i in range(len(nodelist)):
                if parent_name == nodelist[i].name:
                    parent = nodelist[i]
                if children_name == nodelist[i].name:
                    children = nodelist[i]
            if (parent is not None) and (children is not None):
                children.add_parent(parent)
        
        '''
        
        
        
        return None
    
        
    
    def reset_parameters(self):

        #self.model.reset_parameters()
        return None
        
    
    """
    fields_dict = {
                       ('paper', 'to', 'author'):adj_t_dict[('paper', 'to', 'author')], \
                       ('paper', 'to', 'field_of_study'):adj_t_dict[('paper', 'to', 'field_of_study')], \
                       ('paper', 'to', 'citation'):adj_t_dict[('paper', 'to', 'citation')]}
    """

    def forward(self, device, X, Y, adj_t_dict, embedding=None):
        # use message dictionary and layout to build a belief propagation model
        # given a paper, find authors, citations and fields of study, update messages
        pred = []
        #for i in range(X.size(0)):
        for i in range(10):
            paper_id = i
            
            author_list = adj_t_dict[('paper', 'to', 'author')][paper_id].coalesce().indices()
            for i in author_list[0]:
                self.author.likelihood[i] += 1
                
            self.author.likelihood = self.author.likelihood/self.author.likelihood.sum()
           
            
            fields_list = adj_t_dict[('paper', 'to', 'field_of_study')][paper_id].coalesce().indices()
            for i in fields_list[0]:
                self.fields.likelihood[i] += 1
            self.fields.likelihood = self.fields.likelihood/self.fields.likelihood.sum()
            
            citation_list = adj_t_dict[('paper', 'to', 'citation')][paper_id].coalesce().indices()
            for i in citation_list[0]:
                self.citation.likelihood[i] += 1
            self.citation.likelihood = self.citation.likelihood/self.citation.likelihood.sum()
                

            self.citation.message_to_parent(self.topic)
            self.fields.message_to_parent(self.topic)            
            
            self.author.message_to_parent(self.topic)
            self.author.message_to_parent(self.institute)
            pred.append(np.argmax(self.topic.get_belief()))
            
        print('belief pred: {}'.format(pred))
        
        return pred


# In[326]:


m_institution_to_author = data.adj_t_dict[('institution', 'to', 'author')]
m_author_to_paper = data.adj_t_dict[('author', 'writes', 'paper')]
m_citation_to_paper = data.adj_t_dict[('paper', 'cites', 'paper')]
m_paper_to_field = data.adj_t_dict[('paper', 'to', 'field_of_study')]
m_paper_to_topic = data.adj_t_dict[('paper', 'to', 'topic')]



# In[355]:


data.adj_t_dict['topic_to_field_message']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[318]:


## MLP

class MLP(torch.nn.Module):
    def __init__(self,in_channels, out_channels, num_mlp_layers=5, feat_dim=128, emb_dim=1, drop_ratio=0):
        super(MLP, self).__init__()
        self.num_mlp_layers = num_mlp_layers
        self.feat_dim = feat_dim
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
        
    
    def reset_parameters(self):        
        return None
    
    def forward(self, device, X, Y, adj_t_dict):
        
        print('MLP training')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        reg_criterion = torch.nn.L1Loss()
    
        loss_accum = 0
        y_pred = []
        
        self = self.to(device)
        
            
        train_dataset = TensorDataset(X, Y)
        loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers = 1)
        if self.training:
            

            for step, batch in enumerate(tqdm(loader, desc="Iteration")):
                x, y = batch
                x = x.to(device).to(torch.float32)
                y = y.to(device)  


                pred = self.mlp(x).view(-1,)
                optimizer.zero_grad()
                loss = reg_criterion(pred, y)
                loss.backward()
                optimizer.step()
                
                pred = pred.squeeze().detach().cpu()

                y_pred.append(pred)

        
            return y_pred
        else:
            # At inference time, relu is applied to output to ensure positivity
            
            for step, batch in enumerate(tqdm(loader, desc="Iteration")):
                x, y = batch
                x = x.to(device).to(torch.float32)
                y = y.to(device)

                pred = torch.clamp(self.mlp(x).view(-1,), min=0, max=50)
                pred = pred.squeeze().detach().cpu()

                y_pred.append(pred)
       
            return y_pred


# In[319]:



class RGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_types):
        super(RGCNConv, self).__init__()        
        
        self.reg_criterion = torch.nn.L1Loss()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        """
        institution <- author
        topic <- (author, citation, fields of study)
        """
        self.organization_layout = [{'parent':'institution','children':'author'},                               {'parent':'topic', 'children':'author'},                               {'parent':'topic', 'children':'citation'},                               {'parent': 'topic', 'children':'fileds_of_study'}]
        
        # `ModuleDict` does not allow tuples :(
        
        self.rel_lins = ModuleDict({
            f'organization': BeliefPropagation(in_channels, out_channels,self.organization_layout, edge_types)
        })

        self.reset_parameters()

    def reset_parameters(self):
        
                
        for lin in self.rel_lins.values():
            lin.reset_parameters()
            
        """

        for lin in self.root_lins.values():
            lin.reset_parameters()

        """
        
    def forward(self, device, X, Y, adj_t_dict, embedding=None):
        
        print('RGCNconv training')
        self.rel_lins['organization'](device, X, Y, adj_t_dict)
        """ paper
        1. build graph for each node in X
        2. update each node with neighbors
        3. update relations with belief propagation
        """

        """
        fields_dict = {('institution', 'to', 'author'):adj_t_dict[('institution', 'to', 'author')], \
                       ('author', 'to', 'paper'):adj_t_dict[('author', 'to', 'paper')], \
                       ('paper', 'to', 'author'):adj_t_dict[('author', 'to', 'paper')].t(), \
                       ('paper', 'cited_by', 'paper'):adj_t_dict[('paper', 'cited_by', 'paper')], \
                       ('paper', 'to', 'field_of_study'):adj_t_dict[('paper', 'to', 'field_of_study')], \
                       ('paper', 'to', 'topic'):adj_t_dict[('paper', 'to', 'topic')], \
                       ('paper', 'to', 'citation'):adj_t_dict[('paper', 'to', 'citation')]}
        
        organization_out = self.rel_lins['organization'](x_dict, fields_dict)
        
        
        out_dict['organization'] = organization_out
       
        """


        
        return X
        
        


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,adj_t_dict, num_layers=2):
        super(RGCN, self).__init__()


        """
        self.embs = ParameterDict({
            key: Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })        
        
        """
        self.convs = ModuleList()
        I = in_channels
        H = hidden_channels
        O = out_channels
        
        self.convs.append(RGCNConv(I, H,adj_t_dict))
        for i in range(num_layers-1):
            self.convs.append(RGCNConv(H, H,adj_t_dict))
        self.convs.append(MLP(H, O, num_mlp_layers=5, feat_dim=in_channels, emb_dim=O, drop_ratio=0))

        self.reset_parameters()

    def reset_parameters(self):
        """
        for emb in self.embs.values():
            torch.nn.init.xavier_uniform_(emb)
        
        """

        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, device, X, Y, adj_t_dict):
        
        print('start training in RGCN')

        for conv in self.convs[:-1]:
            X = conv(device, X, Y, adj_t_dict)
            
        
        print('done with one layer, relu, dropout and move to next')
        return self.convs[-1](device, X, Y, adj_t_dict)
    
    
        #loader = self.convs[0](loader, device, embedding=None)
        #return loader

        


def train(model, device, X, Y, adj_t_dict):
    model.train()

    optimizer.zero_grad()
    reg_criterion = torch.nn.L1Loss()
    y_pred = model(device, X, Y, adj_t_dict)#.log_softmax(dim=-1)
    
    loss_accum = 0
    
    """
    y_true = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        y_true.extend(y)
        
    
    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)
    y_true = y_true.squeeze().reshape(-1)
    y_pred = y_pred.squeeze().reshape(-1)
    
    print('size true:{} pred:{}'.format(y_true.shape, y_pred.shape))

    valid_dataset = TensorDataset(y_true, y_pred)


    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers = 0)

    
    for step, batch in enumerate(tqdm(valid_loader, desc="Iteration")):
        y_true, y_pred = batch
        
        loss = reg_criterion(y_pred, y_true) # F.nll_loss(y_pred, y_true)
        loss.backward(retain_graph=True)   

        loss_accum += loss.item()

    return loss_accum / (step + 1)
    """
    return loss_accum
    
    



@torch.no_grad()
def test(model, device, X,Y, adj_t_dict, evaluator):
    model.eval()
    y_true = Y
    y_pred = model(device, X,Y, adj_t_dict)
    #y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)
    y_true = y_true.reshape(-1,1)
    y_pred = y_pred.reshape(-1,1)
    return [y_true, y_pred]


# In[320]:


device_id = 0

device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


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
    
    # adj = SparseTensor(row=row, col=col)[:sizes[0], :sizes[1]] # TEST
    """
    if keys[2] == 'institution':
        institution_author = adj.t()
    if keys[2] == 'field_of_study':
        paper_fields = adj
    if keys[1] == 'cites':
        citation_paper = adj.t()
    if keys[0] == 'author' and keys[2] == 'paper':
        author_paper = adj
    
    
    
    """

    if keys[0] != keys[2]:
        data.adj_t_dict[keys] = adj.t()
        data.adj_t_dict[(keys[2], 'to', keys[0])] = adj
    else:
        data.adj_t_dict[keys] = adj.to_symmetric()
        


# In[ ]:





# In[321]:



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


# In[ ]:





# In[462]:


# topic to author message
author_paper = data['edge_index_dict'][('author', 'writes', 'paper')].t()
paper_topic = data['y_dict']['paper']

batch_size = data['y_dict']['paper'].size(0)
num_batch = data['num_nodes_dict']['author']
total_size = author_paper.size(0)

topic_author = []
for i in range(num_batch):
    if i * batch_size < total_size:
        abatch = author_paper[i*batch_size: (i+1)*batch_size]
        df_author_paper = pd.DataFrame(abatch.numpy())
        df_paper_topic = pd.DataFrame(paper_topic.numpy())
        df_paper_topic = df_paper_topic.reset_index()

        df_author_topic = pd.merge(df_author_paper, df_paper_topic, left_on=1, right_on='index')
        df_author_topic = df_author_topic.drop([1,'index'], axis=1)
        df_topic_author = df_author_topic[['0_y','0_x']]
        partial_topic_author = torch.Tensor(np.array(df_topic_author))
        
        topic_author.append(partial_topic_author)

join_topic_author = torch.cat(topic_author)        

df_topic_author = pd.DataFrame(join_topic_author.numpy())
df_topic_author = df_topic_author.sort_values(by=[0,1])
df_topic_author_grouped = df_topic_author.groupby([0,1])[1].count()
df_topic_author_grouped = pd.DataFrame(df_topic_author_grouped)
df_topic_author_grouped = df_topic_author_grouped.rename(columns={1:'count'})
df_topic_author_grouped = df_topic_author_grouped.reset_index()
df_topic_author_sum = df_topic_author_grouped.groupby([0])['count'].sum()
df_topic_author_sum = pd.DataFrame(df_topic_author_sum)
df_topic_author_sum = df_topic_author_sum.rename(columns={'count':'sum'})
df_topic_author_sum = df_topic_author_sum.reset_index()
df_topic_author_prob = pd.merge(df_topic_author_grouped, df_topic_author_sum, on=0)
df_topic_author_prob['prob'] = df_topic_author_prob['count']*1.0/df_topic_author_prob['sum']
df_topic_author_prob = df_topic_author_prob.drop(['count','sum'],axis=1)

df_topic_author_prob[0] = df_topic_author_prob[0].apply(lambda s: int(s))
df_topic_author_prob[1] = df_topic_author_prob[1].apply(lambda s: int(s))


institute_author =  data['edge_index_dict'][('author', 'affiliated_with', 'institution')].t().numpy()
df_institute_author = pd.DataFrame(institute_author, columns=['author','institute'])
df_institute_topic_author = pd.merge(df_institute_author, df_topic_author_prob, left_on='author', right_on=1)
df_institute_topic_author = df_institute_topic_author.rename(columns={0:'topic'})
df_institute_topic_author = df_institute_topic_author.drop([1], axis=1)

df_institute_topic_author = df_institute_topic_author.rename(columns={'institute':0,'topic':1, 'author':2})
institute_topic_to_author_message = df_institute_topic_author


# In[514]:


institute_topic_to_author_message[2].max()


# In[ ]:





# In[ ]:





# In[ ]:





# In[460]:


# topic to citation message

citation = torch.cat((data['edge_index_dict'][('paper', 'cites', 'paper')][1].reshape(-1,1),data['edge_index_dict'][('paper', 'cites', 'paper')][0].reshape(-1,1)),1)


author_paper = citation
paper_topic = data['y_dict']['paper']

batch_size = data['y_dict']['paper'].size(0)
num_batch = data['num_nodes_dict']['author']
total_size = author_paper.size(0)

topic_author = []
for i in range(num_batch):
    if i * batch_size < total_size:
        abatch = author_paper[i*batch_size: (i+1)*batch_size]
        df_author_paper = pd.DataFrame(abatch.numpy())
        df_paper_topic = pd.DataFrame(paper_topic.numpy())
        df_paper_topic = df_paper_topic.reset_index()

        df_author_topic = pd.merge(df_author_paper, df_paper_topic, left_on=1, right_on='index')
        df_author_topic = df_author_topic.drop([1,'index'], axis=1)
        df_topic_author = df_author_topic[['0_y','0_x']]
        partial_topic_author = torch.Tensor(np.array(df_topic_author))
        
        topic_author.append(partial_topic_author)

join_topic_author = torch.cat(topic_author)        


df_topic_author = pd.DataFrame(join_topic_author.numpy())
df_topic_author = df_topic_author.sort_values(by=[0,1])
df_topic_author_grouped = df_topic_author.groupby([0,1])[1].count()
df_topic_author_grouped = pd.DataFrame(df_topic_author_grouped)
df_topic_author_grouped = df_topic_author_grouped.rename(columns={1:'count'})
df_topic_author_grouped = df_topic_author_grouped.reset_index()
df_topic_author_sum = df_topic_author_grouped.groupby([0])['count'].sum()
df_topic_author_sum = pd.DataFrame(df_topic_author_sum)
df_topic_author_sum = df_topic_author_sum.rename(columns={'count':'sum'})
df_topic_author_sum = df_topic_author_sum.reset_index()
df_topic_author_prob = pd.merge(df_topic_author_grouped, df_topic_author_sum, on=0)

df_topic_author_prob['prob'] = df_topic_author_prob['count']*1.0/df_topic_author_prob['sum']
df_topic_author_prob = df_topic_author_prob.drop(['count','sum'],axis=1)
df_topic_author_prob[0] = df_topic_author_prob[0].apply(lambda s: int(s))
df_topic_author_prob[1] = df_topic_author_prob[1].apply(lambda s: int(s))

m = np.zeros((df_topic_author_prob[0].max()+1, df_topic_author_prob[1].max()+1)) 
for index,row in df_topic_author_prob.iterrows():
    m[int(row[0]),int(row[1])] = row['prob']
    
topic_to_citation_message = df_topic_author_prob


# In[461]:


# topic to field of study message
fields = torch.cat((data['edge_index_dict'][('paper', 'has_topic', 'field_of_study')][1].reshape(-1,1),data['edge_index_dict'][('paper', 'has_topic', 'field_of_study')][0].reshape(-1,1)),1)
author_paper = fields
paper_topic = data['y_dict']['paper']

batch_size = data['y_dict']['paper'].size(0)
num_batch = data['num_nodes_dict']['author']
total_size = author_paper.size(0)

topic_author = []
for i in range(num_batch):
    if i * batch_size < total_size:
        abatch = author_paper[i*batch_size: (i+1)*batch_size]
        df_author_paper = pd.DataFrame(abatch.numpy())
        df_paper_topic = pd.DataFrame(paper_topic.numpy())
        df_paper_topic = df_paper_topic.reset_index()

        df_author_topic = pd.merge(df_author_paper, df_paper_topic, left_on=1, right_on='index')
        df_author_topic = df_author_topic.drop([1,'index'], axis=1)
        df_topic_author = df_author_topic[['0_y','0_x']]
        partial_topic_author = torch.Tensor(np.array(df_topic_author))
        
        topic_author.append(partial_topic_author)

join_topic_author = torch.cat(topic_author)        


df_topic_author = pd.DataFrame(join_topic_author.numpy())
df_topic_author = df_topic_author.sort_values(by=[0,1])
df_topic_author_grouped = df_topic_author.groupby([0,1])[1].count()
df_topic_author_grouped = pd.DataFrame(df_topic_author_grouped)
df_topic_author_grouped = df_topic_author_grouped.rename(columns={1:'count'})
df_topic_author_grouped = df_topic_author_grouped.reset_index()
df_topic_author_sum = df_topic_author_grouped.groupby([0])['count'].sum()
df_topic_author_sum = pd.DataFrame(df_topic_author_sum)
df_topic_author_sum = df_topic_author_sum.rename(columns={'count':'sum'})
df_topic_author_sum = df_topic_author_sum.reset_index()
df_topic_author_prob = pd.merge(df_topic_author_grouped, df_topic_author_sum, on=0)

df_topic_author_prob['prob'] = df_topic_author_prob['count']*1.0/df_topic_author_prob['sum']
df_topic_author_prob = df_topic_author_prob.drop(['count','sum'],axis=1)
df_topic_author_prob[0] = df_topic_author_prob[0].apply(lambda s: int(s))
df_topic_author_prob[1] = df_topic_author_prob[1].apply(lambda s: int(s))

m = np.zeros((df_topic_author_prob[0].max()+1, df_topic_author_prob[1].max()+1)) 
for index,row in df_topic_author_prob.iterrows():
    m[int(row[0]),int(row[1])] = row['prob']
    
topic_to_field_message = df_topic_author_prob


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[505]:


data.adj_t_dict['topic_to_field_message'] = topic_to_field_message

data.adj_t_dict['topic_to_citation_message'] = topic_to_citation_message

data.adj_t_dict['topic_to_author_message'] = institute_topic_to_author_message


# In[538]:


# run train and test

X, Y = data['x_dict']['paper'], data.y_dict['paper']

"""
train_dataset = TensorDataset(X, Y)

valid_dataset = train_dataset
test_dataset = train_dataset

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers = 1)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers = 1)

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers = 1)

"""

root = 'data/ogbn_mag'
os.makedirs(osp.join(root, f'checkpoint'), exist_ok = True)

in_channels = 128
hidden_channels = 128
out_channels = 1 # data.adj_t_dict[('paper', 'to', 'field_of_study')].size(1)
model = RGCN(in_channels, hidden_channels, out_channels,data.adj_t_dict)

evaluator = Evaluator(name='ogbn-mag')

optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, device, X,Y,data.adj_t_dict)
y_true, y_pred = test(model, device, X,Y,data.adj_t_dict, evaluator)


# In[ ]:


acc_list = []
is_labeled = []
correct = []
for i in range(y_true.shape[1]):
    is_labeled = y_true[:,i]==y_pred[:,i]
    correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
    if len(correct) > 0:
        acc_list.append(float(correct.sum()/len(correct)))
len(acc_list)


# In[523]:


m = torch.zeros(8740, 1134649).to_sparse()


# ## Us MLP Separately

# In[ ]:


## MLP

class MLP(torch.nn.Module):
    def __init__(self, num_mlp_layers=5, feat_dim=128, emb_dim=1, drop_ratio=0):
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
        
    
    def reset_parameters(self):        
        return None
    
    def forward(self, loader, device):
        #output = self.mlp(x)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        reg_criterion = torch.nn.L1Loss()
        loss_accum = 0
        y_pred = []
        if self.training:            

            for step, batch in enumerate(tqdm(loader, desc="Iteration")):
                x, y = batch
                x = x.to(device).to(torch.float32)
                y = y.to(device)

                pred = self.mlp(x).view(-1,)
                optimizer.zero_grad()
                loss = reg_criterion(pred, y)
                loss.backward()
                optimizer.step()

                y_pred.append(pred.squeeze().detach().cpu())

        
            return y_pred
        else:
            # At inference time, relu is applied to output to ensure positivity
            
            for step, batch in enumerate(tqdm(loader, desc="Iteration")):
                x, y = batch
                x = x.to(device).to(torch.float32)
                y = y.to(device)

                pred = torch.clamp(self.mlp(x).view(-1,), min=0, max=50)
                y_pred.append(pred.detach().cpu())
       
            return y_pred




def train(model, device, loader):
    model.train()
    loss_accum =  model(loader, device)
    
    return loss_accum

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    
    y_pred = model(loader, device)

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        y_true.append(y)

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)
    y_true = y_true.reshape(-1,1)
    y_pred = y_pred.reshape(-1,1)
    print('y_true.ndim={}, y_pred.ndim={}'.format(y_true.ndim, y_pred.ndim))

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)['acc'] # evaluator.eval(input_dict)["mae"]

def test(model, device, loader):
    model.eval()

    y_pred = model(loader, device)

    return y_pred


shared_params = {
    'num_layers': 5,
    'emb_dim': 300,
    'drop_ratio': 0,
    'graph_pooling': 'sum'
}
device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

model = MLP(num_mlp_layers=5, emb_dim=300, drop_ratio=0).to(device)
evaluator = Evaluator(name='ogbn-mag')


# In[ ]:



dataset = PygNodePropPredDataset(name='ogbn-mag',root='data')

data = dataset[0]

root = 'data/ogbn_mag'
X, Y = data['x_dict']['paper'], data.y_dict['paper']


train_dataset = TensorDataset(X, Y)

valid_dataset = train_dataset
test_dataset = train_dataset

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers = 1)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers = 1)

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers = 1)
os.makedirs(osp.join(root, f'checkpoint'), exist_ok = True)


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
    train_mae = train(model, device, train_loader)

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


# In[ ]:





# In[511]:


data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




