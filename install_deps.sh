#!/bin/bash

echo "剂量验证预测系统 - 依赖安装脚本"
echo "================================"

# 检查Python版本
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "Python版本: $python_version"
else
    echo "错误: 未找到Python3"
    exit 1
fi

# 抑制 macOS Tkinter 弃用警告
export TK_SILENCE_DEPRECATION=1

# 检查虚拟环境是否存在
if [ ! -d "app/venv" ]; then
    echo "错误: 虚拟环境不存在"
    echo "请先运行: python app/setup_env.py"
    exit 1
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source app/venv/bin/activate

if [ $? -ne 0 ]; then
    echo "错误: 无法激活虚拟环境"
    exit 1
fi

echo "✓ 虚拟环境已激活"

# 升级pip
echo "升级pip..."
python3 -m pip install --upgrade pip

# 安装依赖包
echo "安装依赖包..."
python3 -m pip install torch numpy pandas openpyxl scikit-learn

echo "依赖安装完成！"
echo "现在可以运行: python3 run_app.py"

# 测试依赖
echo ""
echo "测试依赖..."
python -c "
import sys
try:
    import torch, pandas, numpy, scipy, openpyxl, sklearn
    print('✓ 所有依赖都已安装!')
except ImportError as e:
    print(f'✗ 依赖测试失败: {e}')
    sys.exit(1)
"

echo ""
echo "======================================="
echo "✓ 设置完成!"
echo ""
echo "使用方法:"
echo "1. 激活虚拟环境: source app/activate_env.sh"
echo "2. 运行应用: ./app/run_in_env.sh"
echo "3. 或者手动:"
echo "   export TK_SILENCE_DEPRECATION=1"
echo "   source app/venv/bin/activate"
echo "   python app/main.py" 