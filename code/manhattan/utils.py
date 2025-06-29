from torch_geometric.data import Data
import torch
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp

def build_day_graphst(traffic_day, initial_embeddings, add_self_loops=True):
    num_hours, _, num_regions = traffic_day.shape
    graphs_t = []

    for hour in range(num_hours):
        # 将initial_embeddings转换成tensor
        x = torch.tensor(initial_embeddings, dtype=torch.float)
        
        # 创建边索引和边权重列表
        edge_index = []
        edge_weight = []

        for i in range(num_regions):
            for j in range(i+1,num_regions):
                if i != j:
                    flow_out = traffic_day[hour,  i, j]
                    flow_in = traffic_day[hour,  j, i]
                    
                    if flow_out > 0:
                        edge_index.append((i, j))
                        edge_weight.append(flow_out)
                    
                    if flow_in > 0:
                        edge_index.append((j, i))
                        edge_weight.append(flow_in)
            # 添加自环
            if add_self_loops:
                sl=traffic_day[hour, i, i]
                if sl>0:
                    edge_index.append((i, i))
                    edge_weight.append(sl)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

        # 创建PyTorch Geometric的Data对象
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        graphs_t.append(graph)
    
    return graphs_t

def build_week_graphst(traffic_week, initial_embeddings, add_self_loops=True):
    num_days, _, num_regions = traffic_week.shape
    graphs_t = []

    for day in range(num_days):
        # 将initial_embeddings转换成tensor
        x = torch.tensor(initial_embeddings, dtype=torch.float)
        
        # 创建边索引和边权重列表
        edge_index = []
        edge_weight = []

        for i in range(num_regions):
            for j in range(i+1,num_regions):
                if i != j:
                    flow_out = traffic_week[day,  i, j]
                    flow_in = traffic_week[day,  j, i]
                    
                    if flow_out > 0:
                        edge_index.append((i, j))
                        edge_weight.append(flow_out)
                    
                    if flow_in > 0:
                        edge_index.append((j, i))
                        edge_weight.append(flow_in)
            # 添加自环
            if add_self_loops:
                sl=traffic_week[day, i, i]
                if sl>0:
                    edge_index.append((i, i))
                    edge_weight.append(sl)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            

        # 创建PyTorch Geometric的Data对象
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        graphs_t.append(graph)
    
    return graphs_t

def build_month_graphst(s_similarity,d_similarity,neighbor,importance_k=20):
    s_edge_index = graph_to_COO(s_similarity, importance_k)
    d_edge_index = graph_to_COO(d_similarity, importance_k)
    n_graph = np.eye(180)
    for i in range(len(neighbor)):
        for region in neighbor[i]:
            n_graph[i, region] = 1
            n_graph[region, i] = 1
    n_graph = sp.coo_matrix(n_graph)
    n_edge_index = np.stack((n_graph.row, n_graph.col))
    s_edge_index = torch.tensor(s_edge_index, dtype=torch.long)
    d_edge_index = torch.tensor(d_edge_index, dtype=torch.long)
    n_edge_index = torch.tensor(n_edge_index, dtype=torch.long)
    return [n_edge_index,s_edge_index,d_edge_index]

def day_mob_s_t(traffic_day):
    # 假设 traffic_matrix 是 (24, 180, 180) 形状的矩阵，表示流量数据
    traffic_matrix =traffic_day  # 从文件加载或者其他方式获取

    # 初始化过渡概率矩阵
    mob_matrix_s_to_t = np.zeros_like(traffic_matrix)  # 流出概率矩阵
    mob_matrix_t_to_s = np.zeros_like(traffic_matrix)  # 流入概率矩阵

    # 计算过渡概率矩阵
    for hour in range(24):
        for i in range(180):
            # 流出总量
            total_outflow = np.sum(traffic_matrix[hour, i, :])
            # 流入总量
            total_inflow = np.sum(traffic_matrix[hour, :, i])
            
            # 防止除零错误
            if total_outflow > 0:
                mob_matrix_s_to_t[hour, i, :] = traffic_matrix[hour, i, :] / total_outflow
            if total_inflow > 0:
                mob_matrix_t_to_s[hour, :, i] = traffic_matrix[hour, :, i] / total_inflow

    # 现在 mob_matrix_s_to_t 和 mob_matrix_t_to_s 包含了所有时间点的过渡概率
    return mob_matrix_s_to_t,mob_matrix_t_to_s

def week_mob_s_t(traffic_week):
    traffic_matrix =traffic_week  # 从文件加载或者其他方式获取

    # 初始化过渡概率矩阵
    mob_matrix_s_to_t = np.zeros_like(traffic_matrix)  # 流出概率矩阵
    mob_matrix_t_to_s = np.zeros_like(traffic_matrix)  # 流入概率矩阵

    # 计算过渡概率矩阵
    for day in range(7):
        for i in range(180):
            # 流出总量
            total_outflow = np.sum(traffic_matrix[day, i, :])
            # 流入总量
            total_inflow = np.sum(traffic_matrix[day, :, i])
            
            # 防止除零错误
            if total_outflow > 0:
                mob_matrix_s_to_t[day, i, :] = traffic_matrix[day, i, :] / total_outflow
            if total_inflow > 0:
                mob_matrix_t_to_s[day, :, i] = traffic_matrix[day, :, i] / total_inflow
    return mob_matrix_s_to_t,mob_matrix_t_to_s

def graph_to_COO(similarity, importance_k):
    #将一个表示节点相似度的矩阵转换为一个稀疏表示的边索引矩阵（COO 格式）
    graph = torch.eye(180)

    for i in range(180):
        graph[np.argsort(similarity[:, i])[-importance_k:], i] = 1
        graph[i, np.argsort(similarity[:, i])[-importance_k:]] = 1

    edge_index = sp.coo_matrix(graph)
    edge_index = np.vstack((edge_index.row, edge_index.col))
    return edge_index

def pair_sample(neighbor):
    positive = torch.zeros(180, dtype=torch.long)
    negative = torch.zeros(180, dtype=torch.long)

    for i in range(180):
        region_idx = np.random.randint(len(neighbor[i]))
        pos_region = neighbor[i][region_idx]
        positive[i] = pos_region
    for i in range(180):
        neg_region = np.random.randint(180)
        while neg_region in neighbor[i] or neg_region == i:
            neg_region = np.random.randint(180)
        negative[i] = neg_region
    return positive, negative

def pair_sample_pn(pos,neg,neighbor):
    positive = torch.zeros(180, dtype=torch.long)
    negative = torch.zeros(180, dtype=torch.long)

    for i in range(180):
        region_idx = np.random.randint(len(pos[i]))
        pos_region = pos[i][region_idx]
        positive[i] = pos_region
    for i in range(180):
        neg_region = np.random.randint(180)
        while neg_region in neighbor[i] or neg_region == i:
            neg_region = np.random.randint(180)
        negative[i] = neg_region
    return positive, negative

def pairwise_inner_product(mat_1, mat_2):
    n, m = mat_1.shape  # (180, 144)
    mat_expand = torch.unsqueeze(mat_2, 0)  # (1, 180, 144),
    mat_expand = mat_expand.expand(n, n, m)  # (180, 180, 144),
    mat_expand = mat_expand.permute(1, 0, 2)  # (180, 180, 144),
    inner_prod = torch.mul(mat_expand, mat_1)  # (180, 180, 144), 
    inner_prod = torch.sum(inner_prod, axis=-1)  # (180, 180),
    return inner_prod

def kl_divergence(p, q, epsilon=1e-6):
    p = p + epsilon
    q = q + epsilon
    return torch.sum(p * torch.log(p / q))

def day_series_loss(s_embeddings, t_embeddings, mob_matrix_s_to_t, mob_matrix_t_to_s):

    logits_s_to_t = pairwise_inner_product(s_embeddings, t_embeddings)
    probs_s_to_t = F.softmax(logits_s_to_t, dim=-1)
    #loss_s_to_t = -torch.sum(mob_matrix_s_to_t * torch.log(probs_s_to_t + 1e-6))
    
    # 计算t到s的过渡概率
    logits_t_to_s = pairwise_inner_product(t_embeddings, s_embeddings)
    probs_t_to_s = F.softmax(logits_t_to_s, dim=-1)
    #loss_t_to_s = -torch.sum(mob_matrix_t_to_s * torch.log(probs_t_to_s + 1e-6))
    
    # 总损失是两个方向损失的和
    #loss = loss_s_to_t + loss_t_to_s
    kl_loss_s = kl_divergence(mob_matrix_s_to_t, probs_s_to_t)
    kl_loss_d = kl_divergence(mob_matrix_t_to_s, probs_t_to_s)
    kl_loss = kl_loss_d+kl_loss_s
    
    return kl_loss

def week_series_loss(s_embeddings, t_embeddings, mob_matrix_s_to_t, mob_matrix_t_to_s):
    logits_s_to_t = pairwise_inner_product(s_embeddings, t_embeddings)
    probs_s_to_t = F.softmax(logits_s_to_t, dim=-1)
    #loss_s_to_t = -torch.sum(mob_matrix_s_to_t * torch.log(probs_s_to_t + 1e-6))
    
    # 计算t到s的过渡概率
    logits_t_to_s = pairwise_inner_product(t_embeddings, s_embeddings)
    probs_t_to_s = F.softmax(logits_t_to_s, dim=-1)
    #loss_t_to_s = -torch.sum(mob_matrix_t_to_s * torch.log(probs_t_to_s + 1e-6))

    # 总损失是两个方向损失的和
    kl_loss_s = kl_divergence(mob_matrix_s_to_t, probs_s_to_t)
    kl_loss_d = kl_divergence(mob_matrix_t_to_s, probs_t_to_s)
    kl_loss = kl_loss_d+kl_loss_s
    
    return kl_loss

def month_loss(s_emb, d_emb, mob):
    inner_prod = torch.mm(s_emb, d_emb.T)
    ps_hat = F.softmax(inner_prod, dim=-1)
    inner_prod = torch.mm(d_emb, s_emb.T)
    pd_hat = F.softmax(inner_prod, dim=-1)
    loss = torch.sum(-torch.mul(mob, torch.log(ps_hat)) -
                     torch.mul(mob, torch.log(pd_hat)))
    return loss

