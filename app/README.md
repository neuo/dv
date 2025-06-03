# 剂量验证预测系统

一个基于深度学习的误差预测桌面应用，支持CNN、Transformer和CNN+Transformer混合模型。

## 功能特性

- 🤖 **多模型支持**: CNN、Transformer、CNNTransformer
- 🎯 **多精度配置**: 支持5种不同精度配置（2mm 2%、2mm 3%、3mm 2%、1.5mm 1.5%、1mm 1%）
- 📁 **文件输入**: 支持Excel格式的ADT_plan、Ref_measure、Ref_plan文件
- 🖥️ **跨平台**: 支持Windows、macOS、Linux
- 🎨 **用户友好**: 直观的图形界面，实时预测进度显示

## 系统要求

- Python 3.7+
- 已训练好的模型文件（位于`result/`目录）

## 安装依赖

```bash
pip install -r app/requirements.txt
```

或者手动安装：

```bash
pip install torch pandas numpy scipy openpyxl scikit-learn
```

## 使用方法

### 方法1: 直接运行
```bash
python app/gui_fixed.py
```

### 方法2: 通过主入口运行
```bash
python app/main.py
```

### 方法3: 在虚拟环境中运行
```bash
source app/venv/bin/activate
python app/gui_fixed.py
```

### 方法4: 使用便捷脚本
```bash
./app/run_in_env.sh
```

## 界面说明

### 1. 模型配置
- **选择模型**: 从可用的模型中选择（CNN、Transformer、CNNTransformer）
- **选择精度**: 选择对应的精度配置

### 2. 输入文件
- **ADT_plan**: 自适应计划文件（Excel格式）
- **Ref_measure**: 参考测量文件（Excel格式）
- **Ref_plan**: 参考计划文件（Excel格式）

### 3. 预测操作
- 点击"开始预测"按钮进行预测
- 预测过程中会显示进度条
- 结果将显示在下方的文本区域

## 文件结构

```
app/
├── __init__.py          # 包初始化文件
├── main.py              # 主入口文件
├── gui.py               # GUI界面
├── model_loader.py      # 模型加载器
├── requirements.txt     # 依赖文件
└── README.md           # 说明文档
```

## 模型文件要求

应用需要以下模型文件位于`result/`目录：

```
result/
├── CNN_0.pth           # CNN模型 - 2mm 2%
├── CNN_1.pth           # CNN模型 - 2mm 3%
├── CNN_2.pth           # CNN模型 - 3mm 2%
├── CNN_3.pth           # CNN模型 - 1.5mm 1.5%
├── CNN_4.pth           # CNN模型 - 1mm 1%
├── Transformer_0.pth   # Transformer模型 - 2mm 2%
├── Transformer_1.pth   # Transformer模型 - 2mm 3%
├── Transformer_2.pth   # Transformer模型 - 3mm 2%
├── Transformer_3.pth   # Transformer模型 - 1.5mm 1.5%
├── Transformer_4.pth   # Transformer模型 - 1mm 1%
├── CNNTransformer_0.pth # 混合模型 - 2mm 2%
├── CNNTransformer_1.pth # 混合模型 - 2mm 3%
├── CNNTransformer_2.pth # 混合模型 - 3mm 2%
├── CNNTransformer_3.pth # 混合模型 - 1.5mm 1.5%
└── CNNTransformer_4.pth # 混合模型 - 1mm 1%
```

## 输入文件格式

输入的Excel文件应该包含：

1. **ADT_plan.xlsx**: 自适应计划数据矩阵
2. **Ref_measure.xlsx**: 参考测量数据矩阵  
3. **Ref_plan.xlsx**: 参考计划数据矩阵

文件格式应与训练数据保持一致。

## 故障排除

### 1. 模型文件缺失
```
错误: 没有找到可用的模型文件
```
**解决方案**: 确保`result/`目录下有完整的模型文件

### 2. 依赖包缺失
```
错误: 缺少以下依赖包: torch
```
**解决方案**: 运行 `pip install -r app/requirements.txt`

### 3. 文件格式错误
```
预测失败: 文件必须是Excel格式
```
**解决方案**: 确保输入文件是`.xlsx`或`.xls`格式

### 4. 数据处理错误
```
预测失败: 数据预处理失败
```
**解决方案**: 检查输入文件的数据格式是否正确

## 技术支持

如遇到问题，请检查：

1. Python版本是否为3.7+
2. 所有依赖包是否正确安装
3. 模型文件是否完整
4. 输入文件格式是否正确

## 版本信息

- 版本: 1.0.0
- 支持的操作系统: Windows, macOS, Linux
- Python要求: 3.7+ 