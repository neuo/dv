# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# 导入模型类
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.cnn import CNNModel
from model.transformer import TransformerModel
from model.cnn_transfrom import CNNTransformerModel
from model.cnn_single import CNNSingleModel
from preprocess import process_data_group

class ModelLoader:
    """模型加载和预测管理器"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {
            'CNN': CNNModel,
            'Transformer': TransformerModel,
            'CNNTransformer': CNNTransformerModel,
            'CNN_SINGLE': CNNSingleModel
        }
        
        # 查找模型文件目录
        self.result_dir = self._find_result_directory()
        print(f"模型目录: {self.result_dir}")
        
        self.precision_configs = [
            "2mm 2%",
            "2mm 3%", 
            "3mm 2%",
            "1.5mm 1.5%",
            "1mm 1%"
        ]
    
    def _find_result_directory(self) -> Optional[Path]:
        """查找result目录，支持多种部署环境"""
        possible_paths = []
        
        # 获取当前脚本的目录
        if getattr(sys, 'frozen', False):
            # 如果是打包的可执行文件
            app_dir = Path(sys.executable).parent
        else:
            # 如果是Python脚本
            app_dir = Path(__file__).parent
        
        # 可能的result目录位置
        possible_paths = [
            app_dir / "result",                    # 与可执行文件同目录
            app_dir.parent / "result",             # 上级目录
            Path.cwd() / "result",                 # 当前工作目录
            Path.cwd().parent / "result",          # 当前工作目录的上级
        ]
        
        # 如果在开发环境中，添加更多路径
        if not getattr(sys, 'frozen', False):
            possible_paths.extend([
                Path(__file__).parent.parent / "result",  # 项目根目录
                Path("/Users/neuo/Documents/dv/result"),  # 绝对路径（开发环境）
            ])
        
        # 查找存在的目录
        for path in possible_paths:
            if path.exists() and path.is_dir():
                # 检查是否包含模型文件
                model_files = list(path.glob("*.pth"))
                if model_files:
                    print(f"找到模型目录: {path} (包含 {len(model_files)} 个模型文件)")
                    return path
                else:
                    print(f"目录存在但无模型文件: {path}")
        
        print("警告: 未找到包含模型文件的result目录")
        print("搜索路径:")
        for path in possible_paths:
            print(f"  - {path} {'(存在)' if path.exists() else '(不存在)'}")
        
        return None
    
    def get_available_models(self):
        """获取可用的模型列表"""
        if not self.result_dir:
            print("警告: 模型目录未找到，返回空列表")
            return []
        
        available = []
        for model_name in self.model_configs.keys():
            # 检查是否所有精度配置的模型文件都存在
            all_exist = True
            for i in range(5):
                model_file = self.result_dir / f"{model_name}_{i}.pth"
                if not model_file.exists():
                    all_exist = False
                    break
            if all_exist:
                available.append(model_name)
            else:
                print(f"模型 {model_name} 不完整，缺少某些精度配置文件")
        
        print(f"可用模型: {available}")
        return available
    
    def load_model(self, model_name, precision_index):
        """加载指定模型和精度配置"""
        if not self.result_dir:
            raise RuntimeError("模型目录未找到，无法加载模型")
        
        if model_name not in self.model_configs:
            raise ValueError(f"不支持的模型类型: {model_name}")
        
        if precision_index < 0 or precision_index >= len(self.precision_configs):
            raise ValueError(f"无效的精度索引: {precision_index}")
        
        # 创建模型实例
        model_class = self.model_configs[model_name]
        model = model_class()
        
        # 加载权重
        model_file = self.result_dir / f"{model_name}_{precision_index}.pth"
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        
        try:
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")
    
    def predict_from_files(self, model_name, precision_index, adt_plan_file, ref_measure_file, ref_plan_file):
        """从文件路径进行预测"""
        try:
            # 所有模型都使用统一的输入接口（三个文件）
            if not adt_plan_file or not ref_measure_file or not ref_plan_file:
                raise ValueError("请提供三个文件：ADT_plan、Ref_measure、Ref_plan")
            
            # 创建临时目录结构
            temp_dir = Path("temp_prediction")
            temp_dir.mkdir(exist_ok=True)
            
            # 复制文件到临时目录
            import shutil
            temp_adt = temp_dir / "ADT_plan.xlsx"
            temp_ref_measure = temp_dir / "Ref_measure.xlsx"
            temp_ref_plan = temp_dir / "Ref_plan.xlsx"
            
            shutil.copy2(adt_plan_file, temp_adt)
            shutil.copy2(ref_measure_file, temp_ref_measure)
            shutil.copy2(ref_plan_file, temp_ref_plan)
            
            # 处理数据
            from preprocess import process_data_group
            ref_plan, ref_measure, adt_plan = process_data_group(str(temp_dir))
            
            # 转换为张量
            ref_plan_tensor = torch.tensor(ref_plan, dtype=torch.float32).unsqueeze(0)
            ref_measure_tensor = torch.tensor(ref_measure, dtype=torch.float32).unsqueeze(0)
            adt_plan_tensor = torch.tensor(adt_plan, dtype=torch.float32).unsqueeze(0)
            
            # 加载模型
            model = self.load_model(model_name, precision_index)
            
            # 进行预测 - 所有模型都使用三个参数
            with torch.no_grad():
                prediction = model(ref_plan_tensor, ref_measure_tensor, adt_plan_tensor)
                result = prediction.item()
            
            # 清理临时文件
            shutil.rmtree(temp_dir)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"预测失败: {str(e)}")
    
    def predict_from_data(self, model_name, precision_index, adt_plan_data, ref_measure_data, ref_plan_data):
        """从numpy数组进行预测"""
        try:
            # 所有模型都需要三个数据数组
            if adt_plan_data is None or ref_measure_data is None or ref_plan_data is None:
                raise ValueError("请提供三个数据数组")
            
            # 转换为张量
            ref_plan_tensor = torch.tensor(ref_plan_data, dtype=torch.float32).unsqueeze(0)
            ref_measure_tensor = torch.tensor(ref_measure_data, dtype=torch.float32).unsqueeze(0)
            adt_plan_tensor = torch.tensor(adt_plan_data, dtype=torch.float32).unsqueeze(0)
            
            # 加载模型
            model = self.load_model(model_name, precision_index)
            
            # 进行预测 - 所有模型都使用三个参数
            with torch.no_grad():
                prediction = model(ref_plan_tensor, ref_measure_tensor, adt_plan_tensor)
                result = prediction.item()
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"预测失败: {str(e)}")
    
    def get_precision_configs(self):
        """获取精度配置列表"""
        return self.precision_configs.copy()
    
    def validate_input_files(self, adt_plan_file, ref_measure_file, ref_plan_file, model_name=None):
        """验证输入文件"""
        # 所有模型都需要三个文件
        required_files = [
            ("ADT_plan", adt_plan_file),
            ("Ref_measure", ref_measure_file),
            ("Ref_plan", ref_plan_file)
        ]
        
        for name, file_path in required_files:
            if not file_path:
                return False, f"请选择{name}文件"
            
            path = Path(file_path)
            if not path.exists():
                return False, f"{name}文件不存在: {file_path}"
            
            if not path.suffix.lower() in ['.xlsx', '.xls']:
                return False, f"{name}文件必须是Excel格式"
        
        return True, "文件验证通过" 