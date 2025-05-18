import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  GCNConv 

def graph_to_COO(similarity, importance_k):
    #将一个表示节点相似度的矩阵转换为一个稀疏表示的边索引矩阵（COO 格式）
    graph = torch.eye(194)

    for i in range(194):
        graph[np.argsort(similarity[:, i])[-importance_k:], i] = 1
        graph[i, np.argsort(similarity[:, i])[-importance_k:]] = 1

    edge_index = sp.coo_matrix(graph)
    edge_index = np.vstack((edge_index.row, edge_index.col))
    return edge_index

def create_graph(similarity, importance_k):
    edge_index = graph_to_COO(similarity, importance_k)
    return edge_index

def pair_sample(neighbor):
    positive = torch.zeros(194, dtype=torch.long)
    negative = torch.zeros(194, dtype=torch.long)

    for i in range(194):
        region_idx = np.random.randint(len(neighbor[i]))
        pos_region = neighbor[i][region_idx]
        positive[i] = pos_region
    for i in range(194):
        neg_region = np.random.randint(194)
        while neg_region in neighbor[i] or neg_region == i:
            neg_region = np.random.randint(194)
        negative[i] = neg_region
    return positive, negative

def pair_sample_dtw(pos,neg,neighbor):
    positive = torch.zeros(194, dtype=torch.long)
    negative = torch.zeros(194, dtype=torch.long)

    for i in range(194):
        region_idx = np.random.randint(len(pos[i]))
        pos_region = pos[i][region_idx]
        positive[i] = pos_region
    for i in range(194):
        neg_region = np.random.randint(194)
        while neg_region in neighbor[i] or neg_region == i:
            neg_region = np.random.randint(194)
        negative[i] = neg_region
    return positive, negative

def create_neighbor_graph(neighbor):
    graph = np.eye(194)

    for i in range(len(neighbor)):
        for region in neighbor[i]:
            graph[i, region] = 1
            graph[region, i] = 1
    graph = sp.coo_matrix(graph)
    edge_index = np.stack((graph.row, graph.col))
    return edge_index

class RelationGCN(nn.Module):
    def __init__(self, embedding_size, dropout, gcn_layers):
        super(RelationGCN, self).__init__()
        self.gcn_layers = gcn_layers
        self.dropout = dropout

        self.gcns = nn.ModuleList([GCNConv(in_channels=embedding_size, out_channels=embedding_size)
                                   for _ in range(self.gcn_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(embedding_size)
                                 for _ in range(self.gcn_layers - 1)])
        self.relation_transformation = nn.ModuleList([nn.Linear(embedding_size, embedding_size)
                                                      for _ in range(self.gcn_layers)])

    def forward(self, features, rel_emb, edge_index, is_training=True):
        s_emb = features
        d_emb = features
        n_emb = features
        n_r,s_r, d_r= rel_emb
        n_edge_index,s_edge_index, d_edge_index = edge_index
        for i in range(self.gcn_layers - 1):
            tmp = n_emb
            n_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](torch.multiply(n_emb, n_r), n_edge_index)))
            n_r = self.relation_transformation[i](n_r)
            if is_training:
                n_emb = F.dropout(n_emb, p=self.dropout)

            tmp = s_emb
            s_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](torch.multiply(s_emb, s_r), s_edge_index)))
            s_r = self.relation_transformation[i](s_r)
            if is_training:
                s_emb = F.dropout(s_emb, p=self.dropout)

            tmp = d_emb
            d_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](torch.multiply(d_emb, d_r), d_edge_index)))
            d_r = self.relation_transformation[i](d_r)
            if is_training:
                d_emb = F.dropout(d_emb, p=self.dropout)
            
        n_emb = self.gcns[-1](torch.multiply(n_emb, n_r), n_edge_index)
        s_emb = self.gcns[-1](torch.multiply(s_emb, s_r), s_edge_index)
        d_emb = self.gcns[-1](torch.multiply(d_emb, d_r), d_edge_index)

        n_r = self.relation_transformation[-1](n_r)
        s_r = self.relation_transformation[-1](s_r)
        d_r = self.relation_transformation[-1](d_r)

        return  n_emb, s_emb, d_emb, n_r,  s_r, d_r
    
class CrossLayer(nn.Module):
    def __init__(self, embedding_size):
        super(CrossLayer, self).__init__()
        self.alpha_n = nn.Parameter(torch.tensor(0.3)) #0.4
        self.alpha_d = nn.Parameter(torch.tensor(0.3))
        self.alpha_s = nn.Parameter(torch.tensor(0.3))

        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_size, num_heads=4)

    def forward(self, n_emb, s_emb, d_emb):
        stk_emb = torch.stack((n_emb,s_emb,d_emb))
        fusion, _ = self.attn(stk_emb, stk_emb, stk_emb)

        n_f = fusion[0] * self.alpha_n + (1 - self.alpha_n) * n_emb
        s_f = fusion[1] * self.alpha_s + (1 - self.alpha_s) * s_emb
        d_f = fusion[2] * self.alpha_d + (1 - self.alpha_d) * d_emb

        return n_f, s_f, d_f

class AttentionFusionLayer(nn.Module):
    def __init__(self, embedding_size):
        super(AttentionFusionLayer, self).__init__()
        self.q = nn.Parameter(torch.randn(embedding_size))
        self.fusion_lin = nn.Linear(embedding_size, embedding_size)

    def forward(self, n_f,s_f, d_f):
        n_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(n_f)) * self.q, dim=1))
        s_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(s_f)) * self.q, dim=1))
        d_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(d_f)) * self.q, dim=1))

        w_stk = torch.stack((n_w, s_w, d_w))
        w = torch.log_softmax(w_stk, dim=0)

        region_feature = w[0] * n_f + w[1] * s_f + w[2] * d_f 
        return region_feature
    
class month_model(nn.Module):
    def __init__(self, embedding_size, dropout, gcn_layers):
        super(month_model, self).__init__()

        self.relation_gcns = RelationGCN(embedding_size, dropout, gcn_layers)

        self.cross_layer = CrossLayer(embedding_size)

        self.fusion_layer = AttentionFusionLayer(embedding_size)

    def forward(self, features, rel_emb, edge_index, is_training=True):
        n_emb, s_emb, d_emb, n_r, s_r, d_r = self.relation_gcns(features, rel_emb, edge_index, is_training)
        n_f,s_f, d_f= self.cross_layer(n_emb, s_emb, d_emb)

        region_feature = self.fusion_layer(n_f,s_f, d_f)

        n_f = torch.multiply(region_feature, n_r)
        s_f = torch.multiply(region_feature, s_r)
        d_f = torch.multiply(region_feature, d_r)

        return region_feature, n_f, s_f, d_f
    
    