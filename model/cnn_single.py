# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class CNNSingleModel(nn.Module):
    def __init__(self):
        super(CNNSingleModel, self).__init__()

        self.name = 'CNN_SINGLE'

        # 模型的默认学习率和学习率衰减因子
        self.default_lr = 0.00003
        self.lr_decay_factor = 0.1
        self.lr_decay_loop = 10
        
        # 定义 CNN 层（1 通道输入，输出 3 个浮点数）
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
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
        # 训练数据尺寸：(201, 651) -> 卷积后 (128, 25, 81) = 259200
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 25 * 81, 512),  # 使用训练数据的实际尺寸
            nn.ReLU(),
            nn.Linear(512, 1),  
            nn.Sigmoid()  # 将输出限制在 0 到 1
        )
    
        
    def forward(self, ref_plan, ref_measure, adt_plan):
        """
        前向传播函数
        参数:
            ref_plan: 参考计划数据 (不使用，保持接口一致)
            ref_measure: 参考测量数据 (不使用，保持接口一致)
            adt_plan: ADT计划数据 (实际使用的输入)
        """
        # CNN_SINGLE模型只使用adt_plan数据，忽略ref_plan和ref_measure
        # 这样保持与其他模型相同的接口，但只使用单一输入
        
        # 处理adt_plan的维度
        if adt_plan.dim() == 2:
            x = adt_plan.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        elif adt_plan.dim() == 3:
            x = adt_plan.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        else:
            x = adt_plan  # 假设已经是正确的维度 (B, 1, H, W)
        
        # CNN 特征提取
        x = self.conv_layers(x)
        
        # 全连接层进行回归输出
        x = self.fc_layers(x)
        
        # 将输出范围映射到 [0, 100]
        x = x * 100
        return x.squeeze(-1) 



















# vim: set ts=4 sw=4 sts=4 tw=100 et:
