import os
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numba
import torch_sparse
from torch_sparse import SparseTensor, coalesce, cat
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree, subgraph, remove_self_loops, to_undirected, contains_self_loops, is_undirected, stochastic_blockmodel_graph, k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.data import ClusterData
from torch_geometric.datasets import Planetoid, SNAPDataset, Coauthor, WikipediaNetwork, Reddit, Reddit2
from ogb.nodeproppred import PygNodePropPredDataset
from datetime import datetime, timedelta
from collections import defaultdict


def rand_split(x, ps):
    assert abs(sum(ps) - 1) < 1.0e-10

    shuffled_x = np.random.permutation(x)
    n = len(shuffled_x)
    pr = lambda p: int(np.ceil(p*n))

    cs = np.cumsum([0] + ps)

    return tuple(shuffled_x[pr(cs[i]):pr(cs[i+1])] for i in range(len(ps)))


def process_edge_index(num_nodes, edge_index, edge_attr=None):
    def get_undirected(num_nodes, edge_index, edge_attr):
        row, col = edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_attr = None if (edge_attr is None) else torch.cat([edge_attr, edge_attr], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes, op='max')
        return edge_index, edge_attr

    def sort_edge(num_nodes, edge_index):
        idx = edge_index[0]*num_nodes+edge_index[1]
        sid, perm = idx.sort()
        assert sid.unique_consecutive().shape == sid.shape
        return edge_index[:,perm], perm

    # process edge_attr
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if (edge_index.shape[1] > 0) and (not is_undirected(edge_index)):
        edge_index, edge_attr = get_undirected(num_nodes, edge_index, edge_attr)
    edge_index, od = sort_edge(num_nodes, edge_index)
    _, edge_rv = sort_edge(num_nodes, edge_index.flip(dims=[0]))
    assert torch.all(edge_index[:, edge_rv] == edge_index.flip(dims=[0]))

    return edge_index, (None if edge_attr is None else edge_attr[...,od]), edge_rv


def load_citation(name='Cora', transform=None, split=None):
    data = Planetoid(root='datasets', name=name)[0]
    num_nodes = data.x.shape[0]

    if split is not None:
        assert len(split) == 3
        train_idx, val_idx, test_idx = rand_split(num_nodes, split)
        train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

        data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, train_idx, True)
        data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, val_idx, True)
        data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, test_idx, True)

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    return data if (transform is None) else transform(data)


def load_coauthor(name='CS', transform=None, split=[0.3, 0.2, 0.5]):
    data = Coauthor(root='datasets', name=name)[0]
    num_nodes = data.x.shape[0]

    assert len(split) == 3
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)
    train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, train_idx, True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, val_idx, True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, test_idx, True)

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    return data if (transform is None) else transform(data)


def load_county_facebook(transform=None, split=[0.3, 0.2, 0.5], normalize=True):
    dat = pd.read_csv('datasets/county_facebook/dat.csv')
    adj = pd.read_csv('datasets/county_facebook/adj.csv')

    x = torch.tensor(dat.values[:, 0:9], dtype=torch.float32)
    if normalize:
        x = (x - x.mean(dim=0)) / x.std(dim=0)
    y = torch.tensor(dat.values[:, 9] < dat.values[:, 10], dtype=torch.int64)
    edge_index = torch.transpose(torch.tensor(adj.values), 0, 1)

    data = Data(x=x, y=y, edge_index=edge_index)
    num_nodes = data.x.shape[0]
    assert len(split) == 3
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)
    train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, train_idx, True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, val_idx, True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, test_idx, True)

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    return data if (transform is None) else transform(data)


def load_sexual_interaction(transform=None, split=[0.3, 0.2, 0.5]):
    dat = pd.read_csv('datasets/sexual_interaction/dat.csv', header=None)
    adj = pd.read_csv('datasets/sexual_interaction/adj.csv', header=None)

    y = torch.tensor(dat.values[:, 0], dtype=torch.int64)
    x = torch.tensor(dat.values[:, 1:21], dtype=torch.float32)
    edge_index = torch.transpose(torch.tensor(adj.values), 0, 1)

    data = Data(x=x, y=y, edge_index=edge_index)
    num_nodes = data.x.shape[0]
    assert len(split) == 3
    train_idx, val_idx, test_idx = rand_split(num_nodes, split)
    train_idx, val_idx, test_idx = torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

    data.train_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, train_idx, True)
    data.val_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, val_idx, True)
    data.test_mask = torch.zeros(num_nodes, dtype=bool).scatter_(0, test_idx, True)

    data.edge_index, data.edge_weight, data.edge_rv = process_edge_index(num_nodes, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)

    return data if (transform is None) else transform(data)
