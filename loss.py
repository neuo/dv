import torch
import torch.nn as nn

class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, predicted, target):
        # 计算均值
        mean_pred = torch.mean(predicted)
        mean_target = torch.mean(target)
        
        # 计算分子
        numerator = torch.sum((predicted - mean_pred) * (target - mean_target))
        
        # 计算分母
        denominator = torch.sqrt(torch.sum((predicted - mean_pred) ** 2) * torch.sum((target - mean_target) ** 2))
        
        # 计算皮尔逊相关系数
        r = numerator / (denominator + 1e-6)  # 避免除以 0
        
        # 返回 1 - r 作为损失函数（越小越好）
        loss = 1 - r
        return loss.clamp(min=0, max=2)


# 组合损失函数：包含皮尔逊相关系数和 MAE
class CombinedPearsonMAELoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CombinedPearsonMAELoss, self).__init__()
        self.pearson_loss = PearsonLoss()
        self.mae_loss = nn.L1Loss()  # MAE 损失
        self.alpha = alpha  # 皮尔逊相关系数的权重
        self.beta = beta    # MAE 的权重

    def forward(self, predicted, target):
        # 计算皮尔逊相关系数损失
        pearson_loss = self.pearson_loss(predicted, target)
        
        # 计算 MAE 损失
        mae_loss = self.mae_loss(predicted, target)
        
        # 组合损失
        combined_loss = self.alpha * pearson_loss + self.beta * mae_loss
        return combined_loss

class WeightedMAELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMAELoss, self).__init__()
        self.weights = weights  # 权重向量
    
    def forward(self, predicted, target):
        # 计算绝对误差
        abs_error = torch.abs(predicted - target)
        
        # 对误差应用权重
        weighted_error = abs_error * self.weights
        
        # 返回加权 MAE 损失的平均值
        loss = torch.mean(weighted_error)
        return loss

class WeightedMAELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMAELoss, self).__init__()
        self.weights = weights  # 权重向量
    
    def forward(self, predicted, target):
        # 检查输入维度，确保 weights 的形状与目标值匹配
        if self.weights.shape[0] != target.shape[-1]:
            raise ValueError("weights 的长度必须与目标值的最后一维匹配")
        
        # 计算绝对误差
        abs_error = torch.abs(predicted - target)
        
        # 扩展 weights 以适应批次维度
        weights_expanded = self.weights.expand_as(abs_error)
        
        # 对误差应用权重
        weighted_error = abs_error * weights_expanded
        
        # 返回加权 MAE 损失的平均值
        loss = torch.mean(weighted_error)  # 确保这是标量
        return loss

class MonotonicLoss(nn.Module):
    def __init__(self, base_loss_fn=nn.L1Loss(), alpha=0.5):
        super(MonotonicLoss, self).__init__()
        self.base_loss_fn = base_loss_fn  # 如 MAE 或 MSE
        self.alpha = alpha  # 权重系数，用于控制约束项在总损失中的影响
    
    def forward(self, predicted, target):
        # 计算基础损失（如 MAE）
        base_loss = self.base_loss_fn(predicted, target)
        
        # 计算顺序约束项：输出的相邻元素是否满足递减关系
        monotonicity_penalty = torch.mean(torch.relu(predicted[:, :-1] - predicted[:, 1:]))
        
        # 组合损失
        total_loss = base_loss + self.alpha * monotonicity_penalty
        return total_loss