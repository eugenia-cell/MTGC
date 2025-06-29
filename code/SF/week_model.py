import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  GCNConv 
import math
import random
import numpy as np

# 设置随机种子
seed=2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def get_weekSegment():
    is_weekends=[]
    for time_step in range(7):
        if time_step>=0 and time_step<=4:
            is_weekend=False
        else:
            is_weekend=True 
        is_weekends.append(is_weekend)
    return is_weekends

class TGCNLayer(nn.Module):
    def __init__(self, in_channels, num_time_features,out_channels):
        super(TGCNLayer, self).__init__()
        self.gcn = GCNConv(in_channels+num_time_features, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.leaky_relu = nn.LeakyReLU()
        self.linear = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(0.1)
        self.num_time_features = num_time_features
        self.workday_freq = nn.Parameter(torch.randn(num_time_features))
        self.weekday_freq = nn.Parameter(torch.randn(num_time_features))
        
    def generate_time_encoding(self, time_diff, batch_size, is_weekend):
        time_encoding = []
        for i in range(self.num_time_features):
            if is_weekend:
                frequency = self.workday_freq[i]
                encoding = torch.sin(time_diff * frequency * math.pi)
            else :
                frequency = self.weekday_freq[i]
                encoding = torch.cos(time_diff * frequency * math.pi)
            time_encoding.append(encoding)
        # 将时间编码转换为张量并扩展维度以匹配节点特征维度
        time_encoding = torch.stack(time_encoding, dim=0)  # 维度为 [batch_size, num_time_features]
        time_encoding = time_encoding.unsqueeze(0)  # 扩展维度为 [batch_size, 1, num_time_features]
        time_encoding = time_encoding.repeat(1,batch_size, 1)  # 复制batch维度以匹配节点特征维度
        time_encoding=time_encoding.squeeze()
        return time_encoding
    
    def forward(self, x, edge_index, edge_weight,time_diff,is_weekend):
        #将时空编码添加到节点特征中
        batch_size = x.size(0)
        time_encoding = self.generate_time_encoding(time_diff, batch_size, is_weekend)
        x = torch.cat([x, time_encoding], dim=1)
        # 通过GCN层
        x = self.gcn(x, edge_index, edge_weight)
        
        # 通过BatchNorm层
        x = self.batch_norm(x)
        
        # 通过Leaky ReLU激活函数
        x = self.leaky_relu(x)
        # 通过Linear层
        x = self.linear(x)
        
        # 通过Dropout层
        x = self.dropout(x)
        
        return x

class weekAttention(nn.Module):
    def __init__(self, embed_dim):
        super(weekAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5))
        out = torch.matmul(attention_weights, V)
        
        return out
    
class week_model(nn.Module):
    def __init__(self, in_channels, out_channels,num_time_features):
        super(week_model, self).__init__()
        self.tgn_layer = TGCNLayer(in_channels,num_time_features, out_channels)
        self.num_time_features=num_time_features
        self.attention = weekAttention(out_channels)
        #self.linear_main = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, time_diff,is_weekend):
        # 通过GCN层更新当前时间点的嵌入
        current_embeddings = self.tgn_layer(x, edge_index, edge_weight,time_diff,is_weekend)
        #main_task_output = self.linear_main(current_embeddings)
        return current_embeddings
    
    def combine_week_features(self,week_series_embeddings):
        #overall_integrated_representation = torch.mean(torch.stack(day_series_embeddings), dim=0)
        #return overall_integrated_representation
        workday_mean=sum(week_series_embeddings[0:5])/len(week_series_embeddings[0:5])
        weekday_mean=sum(week_series_embeddings[5:])/len(week_series_embeddings[5:])
        diff_feature=abs(workday_mean-weekday_mean)
        stacked = torch.stack([workday_mean, weekday_mean, diff_feature], dim=1)
        combined = self.attention(stacked)
        # Combine the output of self-attention (180, 96)
        combined = torch.mean(combined, dim=1)
        return combined


