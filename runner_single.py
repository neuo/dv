# -*- coding: utf-8 -*-

# CNN_SINGLE模型训练脚本
# CNN_SINGLE模型使用与其他模型相同的三个输入文件（ADT_plan、Ref_measure、Ref_plan）
# 但在内部只使用ADT_plan数据进行预测，保持接口一致性

# 安装依赖
# pip install pandas numpy scipy openpyxl scikit-learn

import torch.nn as nn
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from loader import init_data_for_train
from check import calculate_errors_numpy, output_result
from process import trainSingle, predictSingle
from model.cnn_single import CNNSingleModel
from loss import CombinedPearsonMAELoss

def main():
    """主训练函数"""
    print("=== CNN_SINGLE模型训练开始 ===")
    print("注意: CNN_SINGLE模型需要三个输入文件，但只使用ADT_plan数据进行预测")
    print("这样保持了与其他模型的接口一致性和数据处理一致性")
    
    # 数据目录
    data_dir = './data/'
    
    # 创建CNN_SINGLE模型实例
    model = CNNSingleModel()
    print(f"模型: {model.name}")
    print(f"学习率: {model.default_lr}")
    print(f"线性层输入尺寸: 128 * 25 * 81 = {128 * 25 * 81}")
    print(f"接口: forward(ref_plan, ref_measure, adt_plan) - 只使用adt_plan")
    
    try:
        # 初始化训练数据
        print("正在加载训练数据...")
        train_datasets, val_datasets, test_datasets = init_data_for_train(data_dir, 0.8, 0.1, 0.1)
        
        # 精度配置描述
        descs = ["2mm 2%", "2mm 3%", "3mm 2%", "1.5mm 1.5%", "1mm 1%"]
        
        # 训练所有精度配置的模型
        for index in range(4, -1, -1):  # 从最高精度开始训练
            print(f"\n--- 训练精度配置 {index}: {descs[index]} ---")
            
            train_dataset = train_datasets[index]
            val_dataset = val_datasets[index]
            test_dataset = test_datasets[index]
            
            # 模型保存路径
            model_path = f"{model.name}_{index}.pth"
            print(f"模型将保存到: {model_path}")
            
            # 选择损失函数
            # criterion = CombinedPearsonMAELoss()  # 组合损失函数
            criterion = nn.MSELoss()  # 简单MSE损失
            print(f"使用损失函数: {criterion.__class__.__name__}")
            
            # 训练模型
            print("开始训练...")
            trainSingle(model, criterion, model_path, train_dataset, val_dataset)
            
            # 预测和评估
            print("开始预测...")
            outputs, targets = predictSingle(model, model_path, test_dataset)
            
            # 输出结果
            print("输出结果...")
            output_result(descs[index], outputs[0], targets[0])
            
            print(f"精度配置 {descs[index]} 训练完成")
        
        print("\n=== 所有模型训练完成 ===")
        print("训练好的模型文件:")
        for i in range(5):
            model_file = f"{model.name}_{i}.pth"
            if Path(model_file).exists():
                print(f"  - {model_file} ({descs[i]})")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 训练成功完成!")
    else:
        print("\n❌ 训练失败!")
        sys.exit(1)


















# vim: set ts=4 sw=4 sts=4 tw=100 et:
