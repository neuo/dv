#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
剂量验证预测系统

一个基于深度学习的误差预测桌面应用
支持CNN、Transformer和CNN+Transformer混合模型

使用方法:
    python app/main.py
    或者
    python -m app.main
"""

import os
import sys
import subprocess
from pathlib import Path

# 抑制 macOS Tkinter 弃用警告
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        ("torch", "torch"),
        ("numpy", "numpy"), 
        ("pandas", "pandas"),
        ("openpyxl", "openpyxl"),
        ("sklearn", "scikit-learn"),  # 注意：导入名和安装包名不同
        ("tkinter", None)  # tkinter通常随Python安装
    ]
    
    missing_packages = []
    
    for import_name, install_name in required_packages:
        if install_name is None:  # 跳过tkinter检查
            continue
            
        try:
            __import__(import_name)
            print(f"✓ {import_name} 已安装")
        except ImportError:
            missing_packages.append(install_name)
            print(f"✗ {import_name} 未安装")
    
    return missing_packages

def install_packages(packages):
    """安装缺失的包"""
    if not packages:
        return True
    
    print(f"\n需要安装以下包: {', '.join(packages)}")
    
    try:
        for package in packages:
            print(f"正在安装 {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ {package} 安装成功")
            else:
                print(f"✗ {package} 安装失败: {result.stderr}")
                return False
        
        return True
    except Exception as e:
        print(f"安装过程中出错: {e}")
        return False

def check_models():
    """检查模型文件是否存在"""
    model_dir = project_root / "result"
    
    if not model_dir.exists():
        print(f"错误: 模型目录不存在: {model_dir}")
        return False
    
    required_models = ["CNN", "Transformer", "CNNTransformer"]
    available_models = []
    
    for model_name in required_models:
        all_exist = True
        for i in range(5):
            model_file = model_dir / f"{model_name}_{i}.pth"
            if not model_file.exists():
                all_exist = False
                break
        if all_exist:
            available_models.append(model_name)
    
    if not available_models:
        print("错误: 没有找到可用的模型文件")
        print(f"请确保在 {model_dir} 目录下有以下模型文件:")
        for model_name in required_models:
            for i in range(5):
                print(f"  - {model_name}_{i}.pth")
        return False
    
    print(f"找到 {len(available_models)} 个可用模型: {', '.join(available_models)}")
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("剂量验证预测系统")
    print("=" * 50)
    
    # 检查依赖
    print("\n1. 检查依赖包...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n发现 {len(missing)} 个缺失的依赖包")
        
        # 询问是否自动安装
        try:
            choice = input("\n是否自动安装缺失的依赖包? (y/n): ").lower().strip()
            if choice in ['y', 'yes', '是']:
                print("\n2. 安装依赖包...")
                if not install_packages(missing):
                    print("\n依赖包安装失败，请手动安装后重试")
                    return False
            else:
                print("\n请手动安装以下依赖包:")
                for package in missing:
                    print(f"  pip install {package}")
                return False
        except KeyboardInterrupt:
            print("\n\n用户取消操作")
            return False
    
    # 检查模型
    print("\n3. 检查模型文件...")
    if not check_models():
        sys.exit(1)
    print("✓ 模型文件检查通过")
    
    # 启动GUI应用
    print("\n4. 启动应用...")
    try:
        from gui_fixed import main as gui_main
        gui_main()
        return True
    except ImportError as e:
        print(f"导入GUI模块失败: {e}")
        print("请确保gui_fixed.py文件存在")
        return False
    except Exception as e:
        print(f"启动应用失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        input("\n按回车键退出...")
        sys.exit(1)
    
    print("剂量验证预测系统") 