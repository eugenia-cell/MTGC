import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tasks import do_tasks

torch.manual_seed(2024)
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_weights = self.log_softmax(torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5))
        out = torch.matmul(attention_weights, V)
        
        return out

class MultiViewContrastiveModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(MultiViewContrastiveModel, self).__init__()
        
        self.encoder_A = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        self.encoder_B = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        self.encoder_C = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
        self.decoder_A = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
        self.decoder_B = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
        self.decoder_C = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
        
        self.self_attention = SelfAttention(embed_dim=output_dim)
    
    def forward(self, A, B, C):
        z_A = self.encoder_A(A)
        z_B = self.encoder_B(B)
        z_C = self.encoder_C(C)
        
        rec_A = self.decoder_A(z_A)
        rec_B = self.decoder_B(z_B)
        rec_C = self.decoder_C(z_C)
        
        return z_A, z_B, z_C, rec_A, rec_B, rec_C

    def get_combined_representation(self, z_A, z_B, z_C):
        # Stack the tensors along a new dimension (180, 3, 96)
        stacked = torch.stack([z_A, z_B, z_C], dim=1)
        
        # Apply self-attention (180, 3, 96)
        combined = self.self_attention(stacked)
        
        # Combine the output of self-attention (180, 96)
        combined = torch.mean(combined, dim=1)
        
        return combined
    
def contrastive_loss(z_i, z_j, temperature=0.01):
    #z_i = F.normalize(z_i, dim=1)
    #z_j = F.normalize(z_j, dim=1)

    similarities = torch.matmul(z_i, z_j.T) / temperature
    labels = torch.arange(len(z_i)).to(z_i.device)

    loss = nn.CrossEntropyLoss()(similarities, labels)
    return loss

def reconstruction_loss(x, rec_x):
    return nn.MSELoss()(x, rec_x)

def multi_view_loss(z_A, z_B, z_C, rec_A, rec_B, rec_C, A, B, C,alpha=0.6):
    loss_AB = contrastive_loss(z_A, z_B)
    loss_AC = contrastive_loss(z_A, z_C)
    loss_BC = contrastive_loss(z_B, z_C)
    
    rec_loss_A = reconstruction_loss(A, rec_A)
    rec_loss_B = reconstruction_loss(B, rec_B)
    rec_loss_C = reconstruction_loss(C, rec_C)
    
    contrastive_loss_avg = (loss_AB + loss_AC + loss_BC) / 3
    reconstruction_loss_avg = (rec_loss_A + rec_loss_B + rec_loss_C) / 3

    return alpha * contrastive_loss_avg + (1 - alpha) * reconstruction_loss_avg
