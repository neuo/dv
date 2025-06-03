#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
打包脚本 - 将应用打包为可执行文件

使用PyInstaller将Python应用打包为独立的可执行文件，
支持Windows (.exe)、macOS (.app)、Linux等平台。

使用方法:
    python app/build_executable.py

依赖:
    pip install pyinstaller
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

def check_pyinstaller():
    """检查PyInstaller是否安装"""
    try:
        import PyInstaller
        print(f"✓ PyInstaller已安装 (版本: {PyInstaller.__version__})")
        return True
    except ImportError:
        print("✗ PyInstaller未安装")
        return False

def install_pyinstaller():
    """安装PyInstaller"""
    print("正在安装PyInstaller...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], 
                      check=True)
        print("✓ PyInstaller安装成功")
        return True
    except subprocess.CalledProcessError:
        print("✗ PyInstaller安装失败")
        return False

def check_dependencies():
    """检查必要的依赖"""
    print("检查依赖...")
    
    required_modules = [
        'torch', 'numpy', 'pandas', 'openpyxl', 
        'sklearn', 'tkinter', 'scipy'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            if module == 'tkinter':
                import tkinter
            elif module == 'sklearn':
                import sklearn
            else:
                __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"缺少依赖: {', '.join(missing_modules)}")
        return False
    
    return True

def build_executable():
    """构建可执行文件"""
    print("开始构建可执行文件...")
    
    # 确定操作系统
    system = platform.system()
    print(f"当前系统: {system}")
    
    # 基础PyInstaller命令参数
    cmd = [
        "pyinstaller",
        "--onefile",                    # 打包成单个文件
        "--name=剂量验证预测系统",        # 可执行文件名称
        "--clean",                      # 清理缓存
    ]
    
    # 根据操作系统添加特定参数
    if system == "Windows":
        cmd.append("--windowed")        # Windows下隐藏控制台
    elif system == "Darwin":  # macOS
        cmd.append("--windowed")        # macOS下创建.app包
    
    # 不打包模型文件到可执行文件中（避免文件过大问题）
    # 模型文件将在安装包中单独提供
    result_dir = Path("../result")
    if result_dir.exists():
        print("✓ 检测到模型文件，将在安装包中单独提供（不打包到可执行文件中）")
    else:
        print("警告: 未找到result目录")
    
    # 添加额外的Python文件
    cmd.extend(["--add-data", f"model_loader.py{os.pathsep}."])
    
    # 添加隐藏导入
    hidden_imports = [
        "torch", "torch.nn", "torch.optim",
        "numpy", "pandas", "openpyxl", 
        "sklearn", "scipy", "scipy.interpolate",
        "tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox",
        "pathlib", "threading", "datetime"
    ]
    
    for module in hidden_imports:
        cmd.extend(["--hidden-import", module])
    
    # 添加主文件
    cmd.append("gui_fixed.py")
    
    try:
        print("执行打包命令...")
        print(f"命令: {' '.join(cmd)}")
        
        # 执行打包命令
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ 可执行文件构建成功")
        
        # 显示输出位置
        dist_dir = Path("dist")
        if dist_dir.exists():
            exe_files = list(dist_dir.glob("*"))
            if exe_files:
                exe_file = exe_files[0]
                file_size = exe_file.stat().st_size / (1024 * 1024)  # MB
                print(f"可执行文件: {exe_file.absolute()}")
                print(f"文件大小: {file_size:.1f} MB")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ 构建失败: {e}")
        if e.stdout:
            print("标准输出:", e.stdout)
        if e.stderr:
            print("错误输出:", e.stderr)
        return False
    except FileNotFoundError:
        print("✗ 未找到pyinstaller命令")
        return False

def create_installer_package():
    """创建安装包"""
    print("创建安装包...")
    
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("✗ 未找到dist目录")
        return False
    
    # 查找可执行文件
    exe_files = list(dist_dir.glob("*"))
    if not exe_files:
        print("✗ 未找到可执行文件")
        return False
    
    exe_file = exe_files[0]
    
    # 创建安装包目录
    package_dir = Path("剂量验证预测系统_安装包")
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    
    # 复制可执行文件
    shutil.copy2(exe_file, package_dir)
    print(f"✓ 复制可执行文件: {exe_file.name}")
    
    # 复制模型文件
    result_dir = Path("../result")
    if result_dir.exists():
        package_result_dir = package_dir / "result"
        shutil.copytree(result_dir, package_result_dir)
        
        # 计算模型文件总大小
        total_size = sum(f.stat().st_size for f in package_result_dir.glob("*.pth"))
        total_size_gb = total_size / (1024 * 1024 * 1024)
        print(f"✓ 复制模型文件: {len(list(package_result_dir.glob('*.pth')))} 个文件, {total_size_gb:.1f} GB")
    else:
        print("警告: 未找到模型文件")
    
    # 创建说明文件
    readme_content = """剂量验证预测系统
==================

这是一个基于深度学习的剂量验证预测系统。

安装说明:
1. 将整个文件夹复制到您希望的位置
2. 双击可执行文件运行即可使用
3. 请保持可执行文件和result文件夹在同一目录下

使用说明:
1. 选择模型类型（CNN、Transformer、CNNTransformer、CNN_SINGLE）
2. 选择精度配置
3. 选择包含数据文件的文件夹（需包含ADT_plan.xlsx、Ref_measure.xlsx、Ref_plan.xlsx）
4. 点击"开始预测"

文件结构:
- 剂量验证预测系统 (可执行文件)
- result/ (模型文件目录，必需)
- 使用说明.txt (本文件)

支持的文件格式:
- Excel文件 (.xlsx, .xls)

系统要求:
- Windows 10/11, macOS 10.14+, 或 Linux
- 至少4GB内存
- 至少3GB可用磁盘空间

注意事项:
- 首次运行可能需要较长时间加载模型
- 请确保有足够的内存运行深度学习模型
- 不要删除或移动result文件夹中的模型文件

技术支持:
如有问题，请联系开发团队。

版本: 1.0
构建时间: {build_time}
"""
    
    from datetime import datetime
    readme_path = package_dir / "使用说明.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content.format(build_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    # 创建快速安装脚本
    if platform.system() == "Windows":
        install_script = """@echo off
echo 剂量验证预测系统快速安装
echo ========================

set "desktop=%USERPROFILE%\\Desktop"
set "target_dir=%desktop%\\剂量验证预测系统"

echo 正在复制到桌面...
if not exist "%target_dir%" mkdir "%target_dir%"
xcopy /E /I /Y "%~dp0*" "%target_dir%\\"

if %errorlevel% == 0 (
    echo ✓ 安装成功！
    echo 程序已安装到桌面的"剂量验证预测系统"文件夹
    echo 您现在可以运行其中的可执行文件
) else (
    echo ✗ 安装失败
)

pause
"""
        script_path = package_dir / "快速安装到桌面.bat"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(install_script)
    
    else:  # macOS/Linux
        install_script = """#!/bin/bash
echo "剂量验证预测系统快速安装"
echo "======================"

desktop="$HOME/Desktop"
target_dir="$desktop/剂量验证预测系统"

echo "正在复制到桌面..."
mkdir -p "$target_dir"
cp -r "$(dirname "$0")"/* "$target_dir/"

if [ $? -eq 0 ]; then
    echo "✓ 安装成功！"
    echo "程序已安装到桌面的'剂量验证预测系统'文件夹"
    echo "您现在可以运行其中的可执行文件"
    chmod +x "$target_dir/剂量验证预测系统"
else
    echo "✗ 安装失败"
fi

read -p "按回车键继续..."
"""
        script_path = package_dir / "快速安装到桌面.sh"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(install_script)
        os.chmod(script_path, 0o755)
    
    # 计算安装包总大小
    total_package_size = sum(f.stat().st_size for f in package_dir.rglob("*") if f.is_file())
    total_package_size_gb = total_package_size / (1024 * 1024 * 1024)
    
    print(f"✓ 安装包创建完成: {package_dir.absolute()}")
    print(f"安装包总大小: {total_package_size_gb:.1f} GB")
    return True

def clean_build_files():
    """清理构建文件"""
    print("清理构建文件...")
    
    dirs_to_remove = ["build", "__pycache__"]
    files_to_remove = ["*.spec"]
    
    for dir_name in dirs_to_remove:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"✓ 删除目录: {dir_name}")
    
    for pattern in files_to_remove:
        for file_path in Path(".").glob(pattern):
            file_path.unlink()
            print(f"✓ 删除文件: {file_path}")

def main():
    """主函数"""
    print("剂量验证预测系统打包工具")
    print("=" * 50)
    
    # 检查当前目录
    if not Path("gui_fixed.py").exists():
        print("✗ 请在app目录下运行此脚本")
        print("使用方法: cd app && python build_executable.py")
        sys.exit(1)
    
    # 检查依赖
    if not check_dependencies():
        print("✗ 请先安装缺少的依赖")
        sys.exit(1)
    
    # 检查PyInstaller
    if not check_pyinstaller():
        if not install_pyinstaller():
            sys.exit(1)
    
    # 构建可执行文件
    if not build_executable():
        sys.exit(1)
    
    # 创建安装包
    if not create_installer_package():
        print("警告: 安装包创建失败，但可执行文件已生成")
    
    # 询问是否清理构建文件
    try:
        choice = input("\n是否清理构建文件? (y/n): ").lower().strip()
        if choice in ['y', 'yes', '是']:
            clean_build_files()
    except KeyboardInterrupt:
        print("\n用户取消操作")
    
    print("\n✓ 打包完成！")
    print("您可以在以下位置找到文件:")
    print("- 可执行文件: dist/")
    print("- 安装包: 剂量验证预测系统_安装包/")

if __name__ == "__main__":
    main() 