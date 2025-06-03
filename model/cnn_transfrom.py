# -*- coding: utf-8 -*-
from datetime import datetime

import torch
import torch.nn as nn


# CNN + Transformer 模型定义
class CNNTransformerModel(nn.Module):
    def __init__(self, num_channels=3, hidden_dim=256, num_heads=8, num_layers=4):
        super(CNNTransformerModel, self).__init__()

        self.name = 'CNNTransformer'

        # 模型的默认学习率和学习率衰减因子
        self.default_lr = 0.00015
        self.lr_decay_factor = 0.2
        self.lr_decay_loop = 10

        # CNN 特征提取部分
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # 将 CNN 的输出展平
        self.flatten = nn.Flatten()
        
        # 全连接层，将 CNN 的展平输出映射到 Transformer 的 hidden_dim (256)
        self.fc_cnn_to_transformer = nn.Linear(518400, hidden_dim)  # 518400 是 CNN 展平后的大小
        
        # Transformer 编码器部分，使用 batch_first=True 以便输入维度为 (batch_size, sequence_length, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 全连接层用于回归预测
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),  # 修改为输出 5 个浮点数
            nn.Sigmoid()
        )

    def forward(self, ref_plan, ref_measure, adt_plan):
        # 将输入合并成 3 通道 (Ref_plan, Ref_measure, ADT_plan)
        x = torch.stack([ref_plan, ref_measure, adt_plan], dim=1)  # 合并到 (B, 3, H, W)
        
        # CNN 特征提取
        cnn_features = self.cnn_layers(x)  # 输出 (B, C, H, W)
        flattened_features = self.flatten(cnn_features)  # 展平为 (B, C*H*W)
        
        # 将 CNN 的输出映射到 Transformer 所需的维度 (B, hidden_dim)
        transformer_input = self.fc_cnn_to_transformer(flattened_features).unsqueeze(1)  # 加一维 (B, 1, hidden_dim)
        
        # Transformer 编码
        transformer_output = self.transformer(transformer_input)  # 输出 (B, 1, hidden_dim)
        
        output = self.fc_out(transformer_output.squeeze(1)) * 100
        return output.squeeze(-1)   # 输出限制在 [0, 100]


# vim: set ts=4 sw=4 sts=4 tw=100 et:
