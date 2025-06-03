# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.name = 'CNN'

        # 模型的默认学习率和学习率衰减因子
        self.default_lr = 0.00003
        self.lr_decay_factor = 0.1
        self.lr_decay_loop = 10
        
        # 定义 CNN 层（3 通道输入，输出 3 个浮点数）
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 池化，尺寸减半
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # 全连接层将特征映射到 5 个输出值
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 25 * 81, 512),  # 假设卷积后输出的尺寸为 (128, 25, 81)
            nn.ReLU(),
            nn.Linear(512, 1),  
            nn.Sigmoid()  # 将输出限制在 0 到 1
        )
    
        
    def forward(self, ref_plan, ref_measure, adt_plan):
        # 将输入的 3 个二维矩阵堆叠为 3 通道的输入
        x = torch.stack([ref_plan, ref_measure, adt_plan], dim=1)  # 合并为 (batch_size, 3, 201, 651)
        
        # CNN 特征提取
        x = self.conv_layers(x)
        
        # 全连接层进行回归输出
        x = self.fc_layers(x)
        
        # 将输出范围映射到 [0, 100]
        x = x * 100
        return x.squeeze(-1) 



















# vim: set ts=4 sw=4 sts=4 tw=100 et:
