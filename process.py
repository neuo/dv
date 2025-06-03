# -*- coding: utf-8 -*-
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from loss import WeightedMAELoss, MonotonicLoss, CombinedPearsonMAELoss


# 训练模型
def train(model, criterion, model_path, train_dataset, val_dataset):
     # 使用模型内置的学习率和衰减因子
    learning_rate = model.default_lr
    lr_decay_factor = model.lr_decay_factor
    lr_decay_loop = model.lr_decay_loop
    
    print("开始准备训练")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    
    # 外部传入
    # criterion = nn.MSELoss()
    # 和 pearson 联合
    # criterion = CombinedPearsonMAELoss(alpha=12, beta=1)
    # 创建加权 MAE 损失函数实例
    # weights = torch.tensor([0.4, 0.5, 0.6, 1.6, 1.6], dtype=torch.float32)  # 权重向量
    # criterion = WeightedMAELoss(weights=weights)
    # criterion = MonotonicLoss(base_loss_fn=nn.L1Loss(), alpha=0.5)

    # 初始化模型
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 定义 StepLR 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_loop, gamma=lr_decay_factor)
    
    # 早停参数
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # 训练模型
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for ref_plan, ref_measure, adt_plan, target in train_dataloader:
            optimizer.zero_grad() # 梯度清零
            # 前向传播
            outputs = model(ref_plan, ref_measure, adt_plan)

            loss = criterion(outputs, target)
            # 反向传播
            loss.backward()
            optimizer.step()
            # 累积损失
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        
        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ref_plan, ref_measure, adt_plan, target in val_dataloader:
                outputs = model(ref_plan, ref_measure, adt_plan)
                loss = criterion(outputs, target)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)

        # 每个 epoch 结束后更新学习率
        scheduler.step()
        
        print(f'${datetime.now()} Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Patience Count: {patience_counter} ')
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存当前最优模型
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停触发，停止训练")
                break


    print("训练完成")

def predict(model, model_path, val_dataset):
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model.load_state_dict(torch.load(model_path))  # 加载模型权重
    model.eval()  # 设置为评估模式

    outputs = [[]]
    targets = [[]]
    # 假设你有一个测试样本 (ref_plan, ref_measure, adt_plan)
    with torch.no_grad():  # 禁用梯度计算，节省内存和加速
        # 准备测试数据 (注意：数据需要符合模型的输入形状)
        # 示例：将测试数据转换为张量 (B, H, W)
        for ref_plan, ref_measure, adt_plan, target in val_dataloader:
        
            # 将测试数据传入模型进行推理
            output = model(ref_plan, ref_measure, adt_plan)

            for index, o in enumerate(output):
                outputs[0].append(o)

                t = target[index]
                targets[0].append(t)

                print(f"模型预测结果: {o}; 实际结果: {t}")
            
    
    return outputs, targets
    

def predict_one(model, model_path, val_dataset):

    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model.load_state_dict(torch.load(model_path))  # 加载模型权重
    model.eval()  # 设置为评估模式

    # 假设你有一个测试样本 (ref_plan, ref_measure, adt_plan)
    with torch.no_grad():  # 禁用梯度计算，节省内存和加速
        # 准备测试数据 (注意：数据需要符合模型的输入形状)
        # 示例：将测试数据转换为张量 (B, H, W)
        for ref_plan, ref_measure, adt_plan, target in val_dataloader:
        
            # 将测试数据传入模型进行推理
            output = model(ref_plan, ref_measure, adt_plan)
            
            print(f"模型预测结果: {output}; 实际结果: {target}")


# 训练模型
def trainSingle(model, criterion, model_path, train_dataset, val_dataset):
     # 使用模型内置的学习率和衰减因子
    learning_rate = model.default_lr
    lr_decay_factor = model.lr_decay_factor
    lr_decay_loop = model.lr_decay_loop
    
    print("开始准备训练")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    
    # 初始化模型
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 定义 StepLR 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_loop, gamma=lr_decay_factor)
    
    # 早停参数
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # 训练模型
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            ref_plan, ref_measure, adt_plan, target = batch[0], batch[1], batch[2], batch[3]
            optimizer.zero_grad() # 梯度清零
            # 前向传播 - 传递三个参数，保持与其他模型一致
            outputs = model(ref_plan, ref_measure, adt_plan)

            loss = criterion(outputs, target)
            # 反向传播
            loss.backward()
            optimizer.step()
            # 累积损失
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        
        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                ref_plan, ref_measure, adt_plan, target = batch[0], batch[1], batch[2], batch[3]
                outputs = model(ref_plan, ref_measure, adt_plan)
                loss = criterion(outputs, target)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)

        # 每个 epoch 结束后更新学习率
        scheduler.step()
        
        print(f'${datetime.now()} Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Patience Count: {patience_counter} ')
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存当前最优模型
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停触发，停止训练")
                break


    print("训练完成")

def predictSingle(model, model_path, val_dataset):
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model.load_state_dict(torch.load(model_path))  # 加载模型权重
    model.eval()  # 设置为评估模式

    outputs = [[]]
    targets = [[]]
    # 假设你有一个测试样本 (ref_plan, ref_measure, adt_plan)
    with torch.no_grad():  # 禁用梯度计算，节省内存和加速
        # 准备测试数据 (注意：数据需要符合模型的输入形状)
        # 示例：将测试数据转换为张量 (B, H, W)
        for batch in val_dataloader:
            ref_plan, ref_measure, adt_plan, target = batch[0], batch[1], batch[2], batch[3]
        
            # 将测试数据传入模型进行推理 - 传递三个参数
            output = model(ref_plan, ref_measure, adt_plan)

            for index, o in enumerate(output):
                outputs[0].append(o)

                t = target[index]
                targets[0].append(t)

                print(f"模型预测结果: {o}; 实际结果: {t}")
            
    
    return outputs, targets



# vim: set ts=4 sw=4 sts=4 tw=100 et:
