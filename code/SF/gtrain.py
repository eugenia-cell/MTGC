import torch
import numpy as np
from day_model import *
from hour_model import *
from month_model import *
from SimCLR import *
from utils import *

# 设置随机种子
np.random.seed(2024)
torch.manual_seed(2024)

# 初始嵌入
initial_embedding = np.random.rand(194, 250) * 2 - 1

# # 加载交通数据
# traffic_data_path = 'Data/train_dataset/2015_07_traffic_matrix.npy'
# traffic_data = np.load(traffic_data_path)

# # 处理交通数据
# traffic_hour = np.sum(traffic_data, axis=1)
# traffic_day = np.sum(traffic_data, axis=0)
# traffic_month = np.sum(traffic_day, axis=0)

traffic_hour=np.load('Data/train_dataset/region_flow_daily.npy')
traffic_day=np.load('Data/train_dataset/region_flow_weekly.npy')
traffic_month=np.sum(traffic_day, axis=0)

mobility = traffic_month.copy()
mobility = mobility / np.mean(mobility)

d_adj = mobility / np.sum(mobility, axis=1, keepdims=True)
d_adj[np.isnan(d_adj)] = 0

s_adj = mobility / np.sum(mobility, axis=0, keepdims=True)
s_adj[np.isnan(s_adj)] = 0

# 将mobility转换为张量，并移动到GPU
mobility = torch.tensor(mobility, dtype=torch.float).cuda()

# 加载邻居数据
neighbor = np.load('Data/train_dataset/SF_neighbors.npy', allow_pickle=True)

# 构建图
hour_graph = build_hour_graphst(traffic_hour, initial_embedding)
day_graph = build_day_graphst(traffic_day, initial_embedding)
month_graph = build_month_graphst(s_adj, d_adj, neighbor)
month_graph = [tensor.cuda() for tensor in month_graph]

hour_segments = get_hourSegment()
day_segments = get_daySegment()

# 初始化特征张量，并移动到GPU
features = torch.randn(194, 144).cuda()
s_r = torch.randn(144).cuda()
d_r = torch.randn(144).cuda()
n_r = torch.randn(144).cuda()
rel_emb = [n_r, s_r, d_r]

# 定义模型参数
node_features = 250
out_channels = 144
embedding_features = 96
h_num_time_features = 4
d_num_time_features = 16
h_time_diff = torch.tensor(1.0).cuda() 
d_time_diff = torch.tensor(24.0).cuda()

# 初始化模型，并移动到GPU
dropout = 0.1
gcn_layers = 3

Dmodel = day_model(node_features, out_channels, h_num_time_features).cuda()
Hmodel = hour_model(node_features, out_channels, d_num_time_features).cuda()
Mmodel = month_model(out_channels, dropout, gcn_layers).cuda()
Multi_view_model = MultiViewContrastiveModel(out_channels, embedding_features, hidden_dim=256).cuda()

# 初始化优化器
optimizerD = optim.Adam(Dmodel.parameters(), lr=0.001, weight_decay=5e-4)
optimizerH = optim.Adam(Hmodel.parameters(), lr=0.001, weight_decay=5e-4)
optimizerM = optim.Adam(Mmodel.parameters(), lr=0.001, weight_decay=5e-4)
optimizerMV = optim.Adam(Multi_view_model.parameters(), lr=0.001, weight_decay=5e-5)

# 定义训练参数
num_epochs = 2000
best_score = 0
best_epoch = 0
best_result = 0, 0, 0, 0

# 处理mobility矩阵
h_mob_matrix_s_to_t, h_mob_matrix_t_to_s = hour_mob_s_t(traffic_hour)
d_mob_matrix_s_to_t, d_mob_matrix_t_to_s = day_mob_s_t(traffic_day)
h_mob_matrix_s_to_t = torch.tensor(h_mob_matrix_s_to_t, dtype=torch.float).cuda()
h_mob_matrix_t_to_s = torch.tensor(h_mob_matrix_t_to_s, dtype=torch.float).cuda()
d_mob_matrix_s_to_t = torch.tensor(d_mob_matrix_s_to_t, dtype=torch.float).cuda()
d_mob_matrix_t_to_s = torch.tensor(d_mob_matrix_t_to_s, dtype=torch.float).cuda()

# 定义损失函数
loss_fn1 = torch.nn.TripletMarginLoss()


# 开始训练
for epoch in range(num_epochs):
    hour_series_embeddings = []
    day_series_embeddings = []
    
    # 训练三个模型
    Dmodel.train()
    Hmodel.train()
    Mmodel.train()

    optimizerD.zero_grad()
    optimizerH.zero_grad()
    optimizerM.zero_grad()
    optimizerMV.zero_grad()
    
    # 获取底层表征 
    for time_step, graph in enumerate(hour_graph):
        optimizerH.zero_grad()
        s_embeddings = Hmodel(graph.x.cuda(), graph.edge_index.cuda(), graph.edge_attr.cuda(), h_time_diff, hour_segments[time_step % 24])
        hour_series_embeddings.append(s_embeddings)
        t_step = time_step + 1
        next_graph = hour_graph[t_step % 24]
        t_embeddings = Hmodel(next_graph.x.cuda(), next_graph.edge_index.cuda(), next_graph.edge_attr.cuda(), h_time_diff, hour_segments[t_step % 24])
        
        hloss = hour_series_loss(s_embeddings, t_embeddings,
                        h_mob_matrix_s_to_t[int(time_step)],
                        h_mob_matrix_t_to_s[int(time_step)])
        hloss.backward(retain_graph=True)
    
    hour_embeddings = Hmodel.combine_hour_features(hour_series_embeddings)

    # 获取中层表征
    for time_step, graph in enumerate(day_graph):
        optimizerD.zero_grad()
        s_embeddings = Dmodel(graph.x.cuda(), graph.edge_index.cuda(), graph.edge_attr.cuda(), d_time_diff, day_segments[time_step % 7])
        day_series_embeddings.append(s_embeddings)
        t_step = time_step + 1
        next_graph = day_graph[t_step % 7]
        t_embeddings = Dmodel(next_graph.x.cuda(), next_graph.edge_index.cuda(), next_graph.edge_attr.cuda(), d_time_diff, day_segments[t_step % 7])
        
        dloss = hour_series_loss(s_embeddings, t_embeddings,
                        d_mob_matrix_s_to_t[int(time_step)],
                        d_mob_matrix_t_to_s[int(time_step)])
        dloss.backward(retain_graph=True)

    day_embeddings = Dmodel.combine_day_features(day_series_embeddings)

    # 获取顶层表征
    optimizerM.zero_grad()
    month_embeddings, n_emb, s_emb, d_emb = Mmodel(features, rel_emb, month_graph)
    m_loss = month_loss(s_emb, d_emb, mobility)
    pos_idx, neg_idx = pair_sample(neighbor)
    geo_loss = loss_fn1(n_emb, n_emb[pos_idx], n_emb[neg_idx])
    mloss = m_loss + geo_loss
    mloss.backward(retain_graph=True)

    # 开始融合
    Multi_view_model.train()
    optimizerMV.zero_grad()
    z_A, z_B, z_C, rec_A, rec_B, rec_C = Multi_view_model(hour_embeddings, day_embeddings, month_embeddings)
    floss = multi_view_loss(z_A, z_B, z_C, rec_A, rec_B, rec_C, hour_embeddings, day_embeddings, month_embeddings)
    floss.backward()
    optimizerH.step()
    optimizerD.step()
    optimizerM.step()
    optimizerMV.step()
    
    with torch.no_grad():
        final_representation = Multi_view_model.get_combined_representation(z_A, z_B, z_C).cpu().numpy()
        results = do_tasks(final_representation, display=False)
        score = results[2] + results[5] + results[6]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_result = results
            # 保存最佳模型参数
            torch.save(Dmodel.state_dict(), 'best_models/Dmodel.pth')
            torch.save(Hmodel.state_dict(), 'best_models/Hmodel.pth')
            torch.save(Mmodel.state_dict(), 'best_models/Mmodel.pth')
            torch.save(Multi_view_model.state_dict(), 'best_models/Multi_view_model.pth')
            np.save('embed/final_embeddings.npy', final_representation)
        print(epoch, results[2], results[5], results[6], results[7], floss)
