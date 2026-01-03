# Model/Diffusion_Model/condition_network.py

import torch
import torch.nn as nn
import numpy as np # Might be needed if adding type hints or defaults

class EnhancedConditionNetwork(nn.Module):
    """
    增强条件网络 - 让国家ID和指数ID的权重显著大于其他5个条件
    (代码与你提供的一致)
    """
    
    def __init__(self, 
                 num_countries=10,      # Default value, should be overridden
                 num_indices=50,       # Default value, should be overridden
                 country_emb_dim=64,     
                 index_emb_dim=128,      
                 numerical_proj_dim=32,  
                 hidden_dim=256,
                 output_dim=128):        # output_dim should match diffusion model's expected cond_dim
        super().__init__()
        
        # === 1. Embedding层 ===
        self.country_embedding = nn.Embedding(num_countries, country_emb_dim)
        self.index_embedding = nn.Embedding(num_indices, index_emb_dim)
        
        nn.init.normal_(self.country_embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self.index_embedding.weight, mean=0, std=0.1)
        
        # === 2. 数值特征投影 ===
        self.numerical_proj = nn.Sequential(
            nn.Linear(5, numerical_proj_dim), # Input is 5 numerical features
            nn.LayerNorm(numerical_proj_dim),
            nn.SiLU()
        )
        
        # === 3. 加权融合层 ===
        total_dim = numerical_proj_dim + country_emb_dim + index_emb_dim
        
        self.numerical_weight = nn.Parameter(torch.tensor(0.8)) 
        self.country_weight = nn.Parameter(torch.tensor(1.0)) # Use float
        self.index_weight = nn.Parameter(torch.tensor(2.0))   # Use float
        
        # === 4. 主干网络 ===
        self.fusion_network = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, output_dim) # Final output dimension
        )
        
    def forward(self, conditions):
        """
        Args:
            conditions: [batch, 7] - Raw condition tensor from DataProcessor
        Returns:
            condition_embedding: [batch, output_dim]
        """
        # === 1. 分离特征 ===
        numerical_features = conditions[:, :5]      
        country_ids = conditions[:, 5].long()       
        index_ids = conditions[:, 6].long()         
        
        # === 2. 获取表示 ===
        numerical_feat = self.numerical_proj(numerical_features) 
        country_emb = self.country_embedding(country_ids)      
        index_emb = self.index_embedding(index_ids)          
        
        # === 3. 加权融合 ===
        # Ensure weights are broadcast correctly if needed (likely okay here)
        weighted_numerical = self.numerical_weight * numerical_feat
        weighted_country = self.country_weight * country_emb
        weighted_index = self.index_weight * index_emb
        
        combined = torch.cat([weighted_numerical, weighted_country, weighted_index], dim=-1)
        
        # === 4. 通过主干网络 ===
        condition_embedding = self.fusion_network(combined)
        
        return condition_embedding
    
    def get_feature_importance(self):
        """返回各特征的相对权重（用于可视化）"""
        # Use .data to get tensor value without gradients
        weights = {
            'numerical': self.numerical_weight.data.item(),
            'country': self.country_weight.data.item(),
            'index': self.index_weight.data.item()
        }
        total = sum(abs(w) for w in weights.values()) # Use abs value for total? Or just sum? Let's use sum.
        if total == 0: return {k: 0 for k in weights}
        return {k: v/total for k, v in weights.items()}