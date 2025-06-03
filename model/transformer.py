# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, num_layers=4):
        super(TransformerModel, self).__init__()

        self.name = 'Transformer'

        # 模型的默认学习率和学习率衰减因子
        self.default_lr = 0.0001
        self.lr_decay_factor = 0.1
        self.lr_decay_loop = 10
        
        # 将二维矩阵展平为序列
        self.flatten = nn.Flatten(start_dim=2)
        
        # 线性变换将 2D 输入映射到 Transformer 的 hidden_dim
        self.fc_in = nn.Linear(201 * 651, hidden_dim)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 全连接层输出 3 个浮点数
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),  # 输出 5 个浮点数
            nn.Sigmoid()  # 将输出限制在 0 到 1
        )
    
    def forward(self, ref_plan, ref_measure, adt_plan):
        # 将输入的 3 个二维矩阵堆叠为 3 通道的输入
        x = torch.stack([ref_plan, ref_measure, adt_plan], dim=1)  # (batch_size, 3, 201, 651)
        
        # 展平为 (batch_size, 3, 201*651)
        x = self.flatten(x)
        
        # 将展平的特征映射到 Transformer 的输入维度
        x = self.fc_in(x)  # 映射到 hidden_dim
        
        # 通过 Transformer 编码器
        x = self.transformer(x)  # 输出形状 (batch_size, 3, hidden_dim)
        
        # 取第一个时间步的输出，作为全连接层输入
        output = self.fc_out(x[:, 0, :])  # 取序列中的第一个位置
        
        # 将输出范围映射到 [0, 100]
        output = output * 100
        return output.squeeze(-1) 


















# vim: set ts=4 sw=4 sts=4 tw=100 et:
