# -*- coding: utf-8 -*-


import numpy as np

def calculate_errors_numpy(predicted, target):
    # 将输入转换为 NumPy 数组
    predicted = np.array(predicted)
    target = np.array(target)

    print("======")
    print(predicted, target)
    print("======")
    
    # 样本数量
    n = predicted.size
    
    # 计算误差
    error = predicted - target
    
    # ME: 平均误差
    me = np.mean(error)
    
    # MAE: 平均绝对误差
    mae = np.mean(np.abs(error))
    
    # RMSE: 均方根误差
    rmse = np.sqrt(np.mean(error ** 2))

    # 皮尔逊相关系数，注意输入至少 2
    pearson_corr = np.corrcoef(predicted, target)[0, 1]
    
    return me, mae, rmse, pearson_corr

def output_result(desc, predicted, target):
    me, mae, rmse, pearson_corr = calculate_errors_numpy(predicted, target)
    print("\n==================")
    print(desc)
    print("原始值:", target)
    print("预测值:", predicted)
    print("ME:", me)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("PEARSON CORR", pearson_corr)
    print(desc)
    print("==================\n")
















# vim: set ts=4 sw=4 sts=4 tw=100 et:
