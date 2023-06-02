#!/usr/bin/env python
# coding: utf-8

# In[1]:



import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter, ModuleDict
from torch_sparse import SparseTensor


from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.data import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
import copy
import argparse

from tqdm import tqdm

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import pandas as pd

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


# In[ ]:


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
    return torch.sparse.mm(convert_to_coo(mat), I.reshape(-1,1))

def init_messages(mat, prior):

    author_coo = convert_to_coo(mat)

    author_values = []
    for i in range(author_coo.size(0)):
        values = author_coo[i]* prior[i].item()
        author_values.append(values.coalesce().values()) 

    values = torch.cat(author_values, 0).reshape(-1)

    author_coo_message = torch.sparse_coo_tensor(author_coo.coalesce().indices(), values,(author_coo.size(0), author_coo.size(1))) 
    return author_coo_message


# In[3]:



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
    
    def forward(self, x_root):
        pred = []
        x_root =x_root.reshape(-1,1)
        for x in x_root:
            #x = x.to(device).to(torch.float32)

            y_pred = self.mlp(x)
            #y_pred = y_pred.detach().cpu().item()
            pred.append(y_pred)

        return pred


# In[236]:


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
        """
        returns marginalized out parent message:
            - in m: group all entries by receiver parent values (all with 0 together, all with 1 together)
            - use other values in groups to get likelihood and messages from other parents
            - multiply those values in each group element
            - sum each group
        """
        likelihood = self.get_likelihood()
        parents_priors = np.array([p.message_to_child(self) for p in self.parents if p != parent])
        parent_i = self.parents.index(parent)
        
        stack = np.array([])
        
        if self.name in ['paper', 'author', 'fields']:
            stack = np.vstack([np.dot(self.m[r].to_dense(), parents_priors.prod(axis=0))                                for r in range(parent.cardinality)])

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
        print('name={} likelihood={}, priors={}'.format(self.name,self.get_likelihood(), self.get_priors()))
        return unnormalized/unnormalized.sum()


class BeliefPropagation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, layout, edge_types):
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
        m_author_to_paper = edge_types[('author', 'to', 'paper')]
        m_citation_to_paper = edge_types[('paper', 'cited_by', 'paper')]
        m_paper_to_field = edge_types[('paper', 'to', 'field_of_study')]
        
        institute = Node("institute")
        institute.cardinality = m_institution_to_author.size(0)
        inst_prior = sum_row(m_institution_to_author)
        inst_prior = inst_prior/torch.sum(inst_prior).item()
        institute.priors = inst_prior.reshape(-1)

        citation = Node("citation")
        citation.cardinality = m_citation_to_paper.size(0)
        citation_prior = sum_row(m_citation_to_paper)
        citation_prior = citation_prior/torch.sum(citation_prior).item()
        citation.priors = citation_prior.reshape(-1)
        
        author = Node('author')
        author.cardinality = m_institution_to_author.size(1)
        author_prior = sum_row(m_institution_to_author)
        author_prior = author_prior.pow(-1)
        author_prior[author_prior.isinf()] = 0
        author.m = init_messages(m_institution_to_author, author_prior)
        
        
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
        
        self.institution = institute
        self.citation = citation
        self.author = author
        self.paper = paper
        self.fields = fields
        
        
        nodelist = [self.institute, self.citation, self.author, self.paper, self.fields]
        
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
        
        return None
    
    """
    fields_dict = {('institution', 'to', 'author'):adj_t_dict[('institution', 'to', 'author')], \
                       ('author', 'to', 'paper'):adj_t_dict[('author', 'to', 'paper')], \
                       ('paper', 'to', 'field_of_study'):adj_t_dict[('paper', 'to', 'field_of_study')], \
                       ('paper', 'cited_by', 'paper'):adj_t_dict[('paper', 'cited_by', 'paper')]}
    """

    def forward(self, x_dict, adj_t_dict, embedding=None):
        # use message dictionary and layout to build a belief propagation model
        
        parent = self.paper.parents[0]
        
        for i in range(len(x_dict['paper'])):
            """
            data.adj_t_dict[('paper', 'to', 'field_of_study')][0]
            data.adj_t_dict[('paper', 'cited_by', 'paper')][0]
            data.adj_t_dict[('author', 'to', 'paper')]
            data.adj_t_dict[('institution', 'to', 'author')]
            
            
            paper_id = i
            field_ids = adj_t_dict[('paper', 'to', 'field_of_study')][paper_id].coalesce().indices()[1]
            citation_ids = adj_t_dict[('paper', 'cited_by', 'paper')][paper_id].t().coalesce().indices()[1]
            author_ids = adj_t_dict[('author', 'to', 'paper')][paper_id].t().coalesce().indices()[1]
        
            """
            
            paper_id = i
            field_ids = convert_to_coo(adj_t_dict[('paper', 'to', 'field_of_study')][paper_id]).coalesce().indices()[1]
            for f_id in field_ids:
                self.fields.likelihood[f_id.item()] +=1
            self.fields.likelihood = self.fields.likelihood/self.fields.likelihood.sum()*self.fields.likelihood.size(0)
        

        
        self.fields.message_to_parent(paper)
        
        return self.fields.get_belief()


# In[ ]:





# In[234]:



class RGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, node_types, edge_types):
        super(RGCNConv, self).__init__()        
        
        self.reg_criterion = torch.nn.L1Loss()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        """
        institution <- author <- paper <- fields of study
                    citation  <-
        """
        self.organization_layout = [{'parent':'institution','children':'author'},                               {'parent':'author', 'children':'paper'},                               {'parent': 'paper', 'children':'fileds_of_study'}]
        self.citation_layout = [{'parent':'citation', 'children':'paper'},                               {'parent': 'paper', 'children':'fileds_of_study'}]
        
        # `ModuleDict` does not allow tuples :(
        
        self.rel_lins = ModuleDict({
            f'organization': BeliefPropagation(in_channels, out_channels,self.organization_layout, edge_types),
            f'citation': BeliefPropagation(in_channels, out_channels,self.citation_layout, edge_types)
        })
        
        
        """ paper
        self.root_lins = ModuleDict({
            'paper': MLP(num_mlp_layers=5, emb_dim=300, drop_ratio=0)        
        })
        
        """

        #self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins.values():
            lin.reset_parameters()
        for lin in self.root_lins.values():
            lin.reset_parameters()

    def forward(self, x_dict, adj_t_dict, embedding=None):
        """
        1. build graph for each node in X
        2. update each node with neighbors
        2.1 update text
        2.2 update authors
        2.3 update institute
        """
        out_dict = {}
        for key, x in x_dict.items():
            if key == 'paper':
                # return the mean of all paper features with the same topic
                # 1. find neighbors with the same topics
                # 2. find clustering center
                # 3. update x_dict
                out_dict[key] = self.root_lins[key](x)
            else:
                out_dict[key] = self.root_lins[key](x)

        fields_dict = {('institution', 'to', 'author'):adj_t_dict[('institution', 'to', 'author')],                        ('author', 'to', 'paper'):adj_t_dict[('author', 'to', 'paper')],                        ('paper', 'to', 'field_of_study'):adj_t_dict[('paper', 'to', 'field_of_study')],                        ('paper', 'cited_by', 'paper'):adj_t_dict[('paper', 'cited_by', 'paper')]}
        
        organization_out = self.rel_lins['organization'](x_dict, adj_t_dict)
        citation_out = self.rel_lins['citation'](x_dict, adj_t_dict)
        
        
        out_dict['organization'] = organization_out
        out_dict['citation'] = citation_out
       

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

        #self.reset_parameters()

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

    out = model(x_dict, adj_t_dict, institution_prob, fields_prob)['paper']
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


# In[7]:


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
#data.node_year_dict = None
#data.edge_reltype_dict = None

print(data)


# In[209]:


data.adj_t_dict[('paper', 'to', 'field_of_study')][0]
data.adj_t_dict[('paper', 'cited_by', 'paper')][0]
data.adj_t_dict[('author', 'to', 'paper')]
data.adj_t_dict[('institution', 'to', 'author')]


# In[ ]:





# In[ ]:





# In[ ]:





# In[231]:


# Convert to new transposed `SparseTensor` format and add reverse edges.

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
#data.node_year_dict = None
#data.edge_reltype_dict = None


data.adj_t_dict = {}
data.fields_prob = None
data.institution_prob = None
paper_fields = None
institution_author = None
author_paper = None
citation_paper = None
for keys, (row, col) in data.edge_index_dict.items():
    sizes = (data.num_nodes_dict[keys[0]], data.num_nodes_dict[keys[2]])
    adj = SparseTensor(row=row, col=col, sparse_sizes=sizes)
    
    # adj = SparseTensor(row=row, col=col)[:sizes[0], :sizes[1]] # TEST
    if keys[2] == 'institution':
        institution_author = adj.t()
    if keys[2] == 'field_of_study':
        paper_fields = adj
    if keys[1] == 'cites':
        citation_paper = adj.t()
    if keys[0] == 'author' and keys[2] == 'paper':
        author_paper = adj
    
    if keys[0] != keys[2]:
        data.adj_t_dict[keys] = adj.t()
        data.adj_t_dict[(keys[2], 'to', keys[0])] = adj
    else:
        data.adj_t_dict[keys] = adj.to_symmetric()
        
data.adj_t_dict[('institution', 'to', 'author')] = institution_author
data.adj_t_dict[('author', 'to', 'paper')] = author_paper
data.adj_t_dict[('paper', 'to', 'field_of_study')] = paper_fields
data.adj_t_dict[('paper', 'cited_by', 'paper')] = citation_paper


# In[78]:


institution_author,author_paper,citation_paper,paper_fields


# In[ ]:



x_types = list(data.x_dict.keys())
edge_types = data.adj_t_dict

model = RGCN(data.x_dict['paper'].size(-1), hidden_channels,
             dataset.num_classes, num_layers, dropout,
             data.num_nodes_dict, x_types, edge_types)

#data = data.to(device)
#model = model.to(device)
train_idx = split_idx['train']['paper']

evaluator = Evaluator(name='ogbn-mag')
logger = Logger(runs)

for run in range(runs):
    #model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, 1 + epochs):
        loss = train(model, data.x_dict, data.adj_t_dict,
                     data.y_dict['paper'], train_idx,  optimizer)
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


# In[79]:





# In[ ]:





# In[ ]:





# In[3]:



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

data.adj_t_dict = {}
for keys, (row, col) in data.edge_index_dict.items():
    sizes = (data.num_nodes_dict[keys[0]], data.num_nodes_dict[keys[2]])
    adj = SparseTensor(row=row, col=col, sparse_sizes=sizes)
    # adj = SparseTensor(row=row, col=col)[:sizes[0], :sizes[1]] # TEST
    if keys[0] != keys[2]:
        data.adj_t_dict[keys] = adj.t()
        #data.adj_t_dict[(keys[2], 'to', keys[0])] = adj
    else:
        data.adj_t_dict[keys] = adj.to_symmetric()
        
print('node types: {}'.format(list(data.x_dict.keys())))
print('edge types: {}'.format(list(data.adj_t_dict.keys())))

x_dict = data.x_dict['paper'].reshape(-1,1,128)


# In[4]:


x_dict.shape


# In[7]:


data.adj_t_dict[('paper', 'cites', 'paper')]


# In[8]:


data.adj_t_dict[('author', 'affiliated_with', 'institution')]


# In[9]:


data.adj_t_dict[('author', 'writes', 'paper')]


# In[10]:


data.adj_t_dict[('paper', 'has_topic', 'field_of_study')]


# In[ ]:





# In[ ]:




