import torch
from torchvision.models import vit_b_16, vit_l_16
from torchvision.models.vision_transformer import VisionTransformer
from torchvision import datasets, transforms
import torch.nn as nn
import math
import torch.nn.functional as F
import copy
from einops import rearrange, repeat
from x_transformers import Encoder, Decoder

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
representation_dim = 512
action_dim = 2

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        img_size, patch_size = (img_size, img_size), (patch_size, patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, X):
        return self.conv(X).flatten(2).transpose(1, 2)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention = AttentionUtil(0.2)
        self.W_q = nn.LazyLinear(embed_dim)
        self.W_k = nn.LazyLinear(embed_dim)
        self.W_v = nn.LazyLinear(embed_dim)
        self.W_o = nn.LazyLinear(embed_dim)
    
    def _transpose1(self, inp):
        inp = inp.reshape(inp.shape[0], inp.shape[1], self.num_heads, -1)
        inp = inp.permute(0, 2, 1, 3)
        return inp.reshape(-1, inp.shape[2], inp.shape[3])
    
    def _transpose2(self, inp):
        inp = inp.reshape(-1, self.num_heads, inp.shape[1], inp.shape[2])
        inp = inp.permute(0, 2, 1, 3)
        return inp.reshape(inp.shape[0], inp.shape[1], -1)

    def forward(self, queries, keys, values):
        queries = self._transpose1(self.W_q(queries))
        keys = self._transpose1(self.W_k(keys))
        values = self._transpose1(self.W_v(values))

        output = self.attention(queries, keys, values)
        output_concat = self._transpose2(output)
        return self.W_o(output_concat)


    
class AttentionUtil(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = FCLayer(mlp_dim, embed_dim, dropout)

    def forward(self, X):
        X = X + self.attention(*([self.ln1(X)] * 3))
        return X + self.mlp(self.ln2(X))

    
class FCLayer(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))


class VisionTransformer_custom(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim,
                 num_layers, num_classes, dropout):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_steps = self.patch_embedding.num_patches + 1 
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", TransformerBlock(
                embed_dim, num_heads, mlp_dim, dropout))
        self.head = nn.Sequential(nn.LayerNorm(embed_dim),
                                  nn.Linear(embed_dim, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for block in self.blks:
            X = block(X)
        return self.head(X[:, 0])
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.vit = vit_b_16().to(device)
        self.vit = VisionTransformer_custom(image_size=65, patch_size=13, in_channels=2,
                                           embed_dim=representation_dim, num_heads=4, mlp_dim=representation_dim,
                                           num_layers=4, num_classes=representation_dim, dropout=0.1).to(device)

    def forward(self, x):
        x = x.to(device)
        # x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False) # [17, 2, 224, 224]
        
        # dummy_channel = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=device)
        # x = torch.cat((x, dummy_channel), dim=1)  # Now [17, 3, 224, 224]
        # print('x loc', x.device)
        
        return self.vit(x)


class Predictor(nn.Module):
    def __init__(self, representation_dim=representation_dim, action_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(representation_dim + action_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, representation_dim)
        ).to(device)
    
    def forward(self, prev_rep, action):
        prev_rep, action = prev_rep.to(device), action.to(device)
        # Concatenate previous representation and action
        input_combined = torch.cat([prev_rep, action], dim=-1).to(device)
        return self.network(input_combined)
    


class JEPAWorldModel(nn.Module):
    """
    Joint Embedding Predictive Architecture World Model with ViT
    """
    def __init__(self, representation_dim=representation_dim, action_dim=2):
        super().__init__()
        self.encoder = VisionTransformer_custom(image_size=65, patch_size=13, in_channels=2,
                                           embed_dim=representation_dim, num_heads=4, mlp_dim=representation_dim,
                                           num_layers=4, num_classes=representation_dim, dropout=0.1).to(device)
        self.predictor = Predictor(representation_dim, action_dim).to(device)
        
        # Use same encoder for target encoder (similar to VicReg)
        self.target_encoder = VisionTransformer_custom(image_size=65, patch_size=13, in_channels=2,
                                           embed_dim=representation_dim, num_heads=4, mlp_dim=representation_dim,
                                           num_layers=4, num_classes=representation_dim, dropout=0.1).to(device)
        
        # Synchronize target encoder with main encoder
        self.update_target_encoder()
    
    def update_target_encoder(self, tau=0.995):
        """
        Exponential Moving Average (EMA) update of target encoder
        """
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * tau + param_q.data * (1. - tau)
    
    # def forward(self, observations, actions):
    #     # Encode observations
    #     observations, actions = observations.to(device), actions.to(device)
    #     encoded_states = [self.encoder(observations[:, 0])]
    #     predicted_states = []
    #     target_states = []
        
    #     # Predict future representations
    #     for t in range(1, observations.shape[1]):
    #         prev_state = encoded_states[-1]
    #         curr_action = actions[:, t-1]

    #         # Predict next state
    #         predicted_state = self.predictor(prev_state, curr_action)
    #         predicted_states.append(predicted_state)
            
    #         # Encode current observation with target encoder
    #         with torch.no_grad():
    #             curr_encoded_state = self.target_encoder(observations[:, t])
    #         target_states.append(curr_encoded_state)

    #         encoded_states.append(self.encoder(observations[:, t]))
        
    #     return predicted_states, target_states


    def forward(self, observations, actions):
        # Move observations and actions to device
        observations, actions = observations.to(device), actions.to(device)
                
        # Encode all observations at once using the encoder
        # encoded_all_states = self.encoder(observations.view(-1, *observations.shape[2:]))
        batch_size, seq_len, channels, height, width = observations.shape
        flat_observations = observations.view(-1, channels, height, width).to(device)
        encoded_all_states = self.encoder(flat_observations).to(device)
        encoded_all_states = encoded_all_states.view(*observations.shape[:2], -1).to(device)  # Reshape back to (batch, sequence, features)
        
        # Initialize storage for predicted and target states
        predicted_states = []
        target_states = []
    
        # Shift actions to align with the sequence (actions at t predict state at t+1)
        prev_states = encoded_all_states[:, :-1]  # Remove the last state
        next_states = encoded_all_states[:, 1:]   # Remove the first state
        # curr_actions = actions[:, :-1]           # Align actions with prediction
        
        # Predict future representations in parallel
        predicted_states = self.predictor(prev_states, actions).to(device)
        
        # Encode target states with target encoder
        with torch.no_grad():
            target_states = self.target_encoder(flat_observations).to(device)  # Skip the first observation for alignment
            target_states = target_states.view(*observations.shape[:2], -1).to(device)
            target_states = target_states[:, 1:]
            target_states = target_states.to(device)
        return predicted_states, target_states
    
    def compute_loss(self, predicted_states, target_states):
        """
        Multi-objective loss to prevent representation collapse
        """
        predicted_states, target_states = predicted_states.to(device), target_states.to(device)
        # 1. Prediction Loss: Minimize distance between predicted and target states
        # pred_loss = F.mse_loss(torch.stack(predicted_states), torch.stack(target_states))
        pred_loss = F.mse_loss(predicted_states, target_states)
        
        # 2. Variance Loss: Encourage representations to have non-zero variance
        std_loss = self.variance_loss(predicted_states)
        
        # 3. Covariance Loss: Decorrelate representation dimensions
        cov_loss = self.covariance_loss(predicted_states)
        
        # Weighted combination of losses
        total_loss = pred_loss + 1e-2 * (std_loss + cov_loss)
        return total_loss
    
    def variance_loss(self, representations, min_std=0.1):
        """Encourage each feature to have non-zero variance"""
        # repr_tensor = torch.stack(representations)
        representations = representations.to(device)
        std_loss = torch.relu(min_std - representations.std(dim=0)).mean()
        return std_loss
    
    def covariance_loss(self, representations):
        """Decorrelate representation dimensions"""
        # repr_tensor = torch.stack(representations)
        representations = representations.to(device)
        repr_tensor = representations
        repr_tensor = repr_tensor.to(device)
        
        # Center the representations
        repr_tensor = repr_tensor - repr_tensor.mean(dim=0)
        
        # Flatten tensor (keep batch dimension intact)
        repr_tensor = repr_tensor.view(repr_tensor.shape[0], -1)
        
        # Compute covariance matrix
        cov_matrix = (repr_tensor.T @ repr_tensor) / (repr_tensor.shape[0] - 1)
        
        # Decorrelate dimensions (set diagonal to zero)
        cov_matrix.fill_diagonal_(0)
        
        # Compute loss
        cov_loss = (cov_matrix ** 2).sum()
        return cov_loss

class DataTransforms:
    """
    Image augmentations and preprocessing for JEPA training
    """
    @staticmethod
    def get_train_transforms():
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

def train_jepa_model(model, dataloader, optimizer, device, epoch):
    """
    Training loop for JEPA world model
    """
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        batch_observations = batch.states.to(device)
        batch_actions = batch.actions.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predicted_states, target_states = model(batch_observations, batch_actions)
        
        # Compute loss
        loss = model.compute_loss(predicted_states, target_states)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update target encoder (EMA)
        model.update_target_encoder()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)



