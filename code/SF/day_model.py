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

def get_daySegment():
    time_segments=[]
    for time_step in range(24):
        if time_step>=6 and time_step<9:
            time_segment='morning_peak'
        elif time_step>=16 and time_step<19:
            time_segment='evening_peak'
        elif time_step>=19 and time_step<24 or time_step>=0 and time_step<6:
            time_segment='nighttime'
        else:
            time_segment='daytime'
        time_segments.append(time_segment)
    return time_segments

class TGCNLayer(nn.Module):
    def __init__(self, in_channels, num_time_features,out_channels):
        super(TGCNLayer, self).__init__()
        self.gcn = GCNConv(in_channels+num_time_features, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.leaky_relu = nn.LeakyReLU()
        self.linear = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(0.1)
        self.num_time_features = num_time_features
        self.morning_peak_freq = nn.Parameter(torch.randn(num_time_features))
        self.evening_peak_freq = nn.Parameter(torch.randn(num_time_features))
        self.nighttime_freq = nn.Parameter(torch.randn(num_time_features))
        self.other_times_freq = nn.Parameter(torch.randn(num_time_features))
        
    def generate_time_encoding(self,time_diff, batch_size,time_segment):
        time_encoding = []
        for i in range(self.num_time_features):
            if time_segment == 'morning_peak':
                frequency = self.morning_peak_freq[i]
                encoding = torch.sin(time_diff * frequency * math.pi)
            elif time_segment == 'evening_peak':
                frequency = self.evening_peak_freq[i]
                encoding = torch.cos(time_diff * frequency * math.pi)
            elif time_segment == 'nighttime':
                frequency = self.nighttime_freq[i]
                encoding = torch.tanh(time_diff * frequency * math.pi)
            else:  # 'other_times'
                frequency = self.other_times_freq[i]
                encoding = (time_diff * frequency * math.pi) / 24.0
            time_encoding.append(encoding)
        time_encoding = torch.stack(time_encoding, dim=0)
        time_encoding = time_encoding.unsqueeze(0)  # 扩展维度为 [batch_size, 1, num_time_features]
        time_encoding = time_encoding.repeat(1,batch_size, 1)  # 复制batch维度以匹配节点特征维度
        time_encoding=time_encoding.squeeze() 
        return time_encoding
    
    def forward(self, x, edge_index, edge_weight,time_diff,time_segment):
        #将时空编码添加到节点特征中
        batch_size = x.size(0)
        time_encoding = self.generate_time_encoding(time_diff, batch_size,time_segment)

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

class DifferentialAttentionFusion(nn.Module):
    def __init__(self, feature_dim=144, alpha=0.4):
        super(DifferentialAttentionFusion, self).__init__()
        self.alpha = alpha
        self.W_q = nn.Parameter(torch.rand(feature_dim, feature_dim))
        self.W_k = nn.Parameter(torch.rand(feature_dim, feature_dim))
        self.W_v = nn.Parameter(torch.rand(feature_dim, feature_dim))

    def forward(self, input_list):
        # 计算差分特征
        diff_list = [input_list[i + 1] - input_list[i] for i in range(len(input_list) - 1)]
        # 初始化综合特征矩阵
        combined_features = torch.zeros((180, 144)).cuda()
        # 遍历每个时间段，计算注意力权重并融合特征
        for i in range(len(diff_list)):
            original_feature = input_list[i]
            diff_feature = diff_list[i]
            # 计算 Query, Key, Value 矩阵
            Q = torch.matmul(original_feature, self.W_q)
            K = torch.matmul(diff_feature, self.W_k)
            V = torch.matmul(diff_feature, self.W_v)
            # 计算注意力权重
            attention_weights = F.softmax(torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32)), dim=-1)
            # 融合特征（加权组合原始特征和差分特征）
            #combined_feature = self.alpha * torch.matmul(attention_weights, V) + (1 - self.alpha) * original_feature
            combined_feature = torch.matmul(attention_weights, original_feature) 
            # 累加得到最终综合特征
            combined_features += combined_feature

        # 取平均以得到最终的综合特征
        combined_features /= len(diff_list)
        return combined_features

class dayAttention(nn.Module):
    def __init__(self, embed_dim):
        super(dayAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5))
        out = torch.matmul(attention_weights, V)
        
        return out
    
class day_model(nn.Module):
    def __init__(self, in_channels, out_channels,num_time_features):
        super(day_model, self).__init__()
        self.tgn_layer = TGCNLayer(in_channels,num_time_features, out_channels)
        self.attention = dayAttention(out_channels)
        self.DIFF_fusion=DifferentialAttentionFusion(out_channels)
        self.num_time_features=num_time_features

    def forward(self, x, edge_index, edge_weight, time_diff,time_segment):
        
        # 通过GCN层更新当前时间点的嵌入
        current_embeddings = self.tgn_layer(x, edge_index, edge_weight,time_diff,time_segment)

        return current_embeddings

    def combine_day_features(self,day_series_embeddings):
        #overall_integrated_representation = torch.mean(torch.stack(day_series_embeddings), dim=0)
        #return overall_integrated_representation
        
        #overall_integrated_representation = self.DIFF_fusion(day_series_embeddings)
        #return overall_integrated_representation
        morningpeak_mean = sum(day_series_embeddings[6:9])/len(day_series_embeddings[6:9])
        eveningpeak_mean = sum(day_series_embeddings[16:19])/len(day_series_embeddings[16:19])
        nightime_mean = sum(day_series_embeddings[19:24])/len(day_series_embeddings[19:24]) + sum(day_series_embeddings[0:6])/len(day_series_embeddings[0:6])
        daytime_mean = sum(day_series_embeddings[9:16])/len(day_series_embeddings[9:16])
        stacked = torch.stack([morningpeak_mean, eveningpeak_mean, nightime_mean,daytime_mean], dim=1)
        combined = self.attention(stacked)
        # Combine the output of self-attention (180, 96)
        combined = torch.mean(combined, dim=1)
        return combined