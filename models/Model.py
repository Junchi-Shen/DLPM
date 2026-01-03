import os
import numpy as np
import torch
import scipy

import torch.nn as nn
import torch.optim as optim
import shutil


from .DiffusionBlocks import DiffusionBlockConditioned
from .Embeddings import SinusoidalPositionalEmbedding

    

#Can predict gaussian_noise, stable_noise, anterior_mean
class MLPModel(nn.Module):
    possible_time_embeddings = [
        'sinusoidal',
        'learnable',
        'one_dimensional_input'
    ]

    def __init__(self, p):
        super(MLPModel, self).__init__()

        # extract from param dict
        self.nfeatures =        p['data']['nfeatures']
        self.use_a_t =          p['model']['use_a_t']
        self.no_a =             p['model']['no_a']
        self.a_pos_emb =        p['model']['a_pos_emb']
        self.a_emb_size =       p['model']['a_emb_size']
        self.time_emb_type =    p['model']['time_emb_type'] 
        self.time_emb_size =    p['model']['time_emb_size']
        self.nblocks =          p['model']['nblocks'] 
        self.nunits =           p['model']['nunits']
        self.skip_connection =  p['model']['skip_connection']
        self.group_norm =       p['model']['group_norm']
        self.dropout_rate =     p['model']['dropout_rate']
        self.device =           p['device']
        self.learn_variance =   p['model']['learn_variance']
        # if isotropic: a_t's are a single dimension, and so is gamma
        self.isotropic =        p[p['method']]['isotropic']
        assert self.no_a and self.isotropic, 'Need to reimplement architecture if model takes non-isotropic a_t as input.'
        # to be computed later depending on chosen architecture
        self.additional_dim =   0 


        assert self.time_emb_type in self.possible_time_embeddings
        
        # for dropout and group norm.
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.group_norm_in = nn.LayerNorm([self.nunits]) if self.group_norm else nn.Identity()
        self.act = nn.SiLU(inplace=False)
        
        # manage time embedding type
        if self.time_emb_type == 'sinusoidal':
            self.time_emb = \
            SinusoidalPositionalEmbedding(self.diffusion_steps, 
                                              self.time_emb_size, self.device)
        elif self.time_emb_type == 'learnable':
            self.time_emb = nn.Linear(1, self.time_emb_size).to(self.device) #Embedding.LearnableEmbedding(1, self.time_emb_size, self.device)
        elif self.time_emb_type == 'one_dimensional_input':
            self.additional_dim += 1
        
        if self.time_emb_type != 'one_dimensional_input':
            # possibly, remove the mlp and just use the embedding
            self.time_mlp = nn.Sequential(self.time_emb,
                                      self.act,
                                      nn.Linear(self.time_emb_size, self.time_emb_size), 
                                      self.act)
        
        # I changed input a_t size from 2 to 1, since we will take it to be equal on both dims
        if self.no_a or self.a_pos_emb:
            self.additional_dim += 0
        elif self.use_a_t:
            self.additional_dim += 1
        else:
            self.additional_dim += 2
            
        # manage a_t embedding
        if self.a_pos_emb:
            dim = 1 if self.use_a_t else 2
            self.a_emb = nn.Linear(dim, self.a_emb_size)
            self.a_mlp = nn.Sequential(self.a_emb,
                                          self.act,
                                          nn.Linear(self.a_emb_size, self.a_emb_size), 
                                          self.act)
        
        # possible exp block, at the end
        
        self.linear_in =  nn.Linear(self.nfeatures + self.additional_dim, self.nunits)
        
        self.inblock = nn.Sequential(self.linear_in,
                                     self.group_norm_in, 
                                     self.act)
        
        self.midblocks = nn.ModuleList([DiffusionBlockConditioned(
                                            self.nunits, 
                                            self.dropout_rate, 
                                            self.skip_connection, 
                                            self.group_norm,
                                            time_emb_size = self.time_emb_size \
                                                if self.time_emb_type != 'one_dimensional_input'\
                                                else False,
                                            a_emb_size = self.a_emb_size\
                                                if self.a_pos_emb \
                                                else False,
                                            activation = nn.SiLU)
                                        for _ in range(self.nblocks)])
        
        # add one conditioned block and one MLP for both mean and variance computation
        self.outblocks_mean = nn.ModuleList([
                DiffusionBlockConditioned(
                                                self.nunits, 
                                                self.dropout_rate, 
                                                self.skip_connection, 
                                                self.group_norm,
                                                time_emb_size = self.time_emb_size \
                                                    if self.time_emb_type != 'one_dimensional_input'\
                                                    else False,
                                                a_emb_size = self.a_emb_size\
                                                    if self.a_pos_emb \
                                                    else False,
                                                activation = nn.SiLU),
                nn.Linear(self.nunits, self.nfeatures)
            ])
        if self.learn_variance:
            self.outblocks_var = nn.ModuleList([
                DiffusionBlockConditioned(
                                                self.nunits, 
                                                self.dropout_rate, 
                                                self.skip_connection, 
                                                self.group_norm,
                                                time_emb_size = self.time_emb_size \
                                                    if self.time_emb_type != 'one_dimensional_input'\
                                                    else False,
                                                a_emb_size = self.a_emb_size\
                                                    if self.a_pos_emb \
                                                    else False,
                                                activation = nn.SiLU),
                nn.Linear(self.nunits, self.nfeatures), #1 if self.isotropic else self.features),
                #nn.Sigmoid()
            ])
        
        

    def forward(self, x, timestep, a_t_prime = None, a_t_1 = None, cond = None):
        ''' g = match_last_dims(self.gammas, x.size())
        bg = match_last_dims(self.bargammas, x.size())

        if a_t_prime is None:
            a_t_prime = torch.ones(size = x.shape).to(self.device)
        if a_t_1 is None:
            a_t_1 = torch.ones(size = x.shape).to(self.device)
        
        a_t = Algo.compute_a_t(g, bg, self.alpha, timestep, a_t_prime, a_t_1)

        # take log of a's to reduce variance
        a_t = torch.log(a_t)
        a_t_prime = torch.log(a_t_prime)
        a_t_1 = torch.log(a_t_1)'''
        
        # inp = [x]
        
        a_t_prime = torch.ones(size = x.shape).to(self.device)
        a_t_1 = torch.ones(size = x.shape).to(self.device)
        a_t = torch.ones(size = x.shape).to(self.device)
        
        # 处理时间步（根据之前的修复）
        if self.time_emb_type == 'one_dimensional_input':
            timestep_expanded = timestep.unsqueeze(1).unsqueeze(2)
            inp += [timestep_expanded]
            t = torch.zeros((x.shape[0], 1, 1)).to(self.device)
        elif self.time_emb_type == 'sinusoidal':
            t = self.time_mlp(timestep.long())
        else:  # learnable
            timestep_reshaped = timestep.unsqueeze(1).to(torch.float32)
            t = self.time_mlp(timestep_reshaped)
        
        # 处理条件信息 X
        if cond is not None:
            # 将条件信息嵌入到时间嵌入中，或者与输入数据拼接
            # 方法1：将条件信息投影后加到时间嵌入
            if not hasattr(self, 'cond_mlp'):
                cond_dim = cond.shape[-1]  # 条件维度
                self.cond_mlp = nn.Sequential(
                    nn.Linear(cond_dim, self.time_emb_size),
                    self.act
                ).to(self.device)
            
            cond_emb = self.cond_mlp(cond)  # [batch_size, time_emb_size]
            t = t + cond_emb  # 将条件信息加到时间嵌入
            
            # 或者方法2：将条件信息与输入数据拼接
            # x = torch.cat([x, cond], dim=-1)  # 需要修改 linear_in 的输入维度
        
        # input
        val = x #torch.hstack(inp)
        # compute
        val = self.inblock(val)
        for midblock in self.midblocks:
            val = midblock(val, t, a)
        
        val_1 = self.outblocks_mean[0](val, t, a)
        val_1 = self.outblocks_mean[1](val_1)
        if self.learn_variance:
            val_2 = self.outblocks_var[0](val, t, a)
            val_2 = self.outblocks_var[1](val_2)
            #val_2 = self.outblocks_var[2](val_2)
            #if self.isotropic:
            #    val_2 = val_2.expand(*val_2.shape[:-1], val_1.shape[-1])
            return torch.concat([val_1, val_2], dim = 1) # concat on channels dim
        else:
            return val_1
            
        # duplicate last
        #val = val.expand(val.shape[0], 2*val.shape[1], *val.shape[2:])

    
    