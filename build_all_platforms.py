#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
跨平台构建指南
==============

由于PyInstaller的限制，每个平台的可执行文件只能在对应平台上构建。

构建方法:

1. Windows平台:
   - 在Windows系统上运行: cd app && python build_executable.py
   - 生成: 剂量验证预测系统.exe

2. macOS平台:
   - 在macOS系统上运行: cd app && python build_executable.py  
   - 生成: 剂量验证预测系统.app

3. Linux平台:
   - 在Linux系统上运行: cd app && python build_executable.py
   - 生成: 剂量验证预测系统 (无扩展名)

自动化构建:
- 使用GitHub Actions: 推送tag到GitHub仓库，自动构建所有平台
- 使用Docker: 在容器中构建Linux版本

注意事项:
- 模型文件(result/)需要单独提供，不包含在可执行文件中
- 每个平台的安装包都包含对应的可执行文件和模型文件
- Windows用户需要.exe文件，macOS用户需要.app文件，Linux用户需要无扩展名的可执行文件
"""

import platform
import sys

def main():
    current_platform = platform.system()
    print(f"当前平台: {current_platform}")
    
    if current_platform == "Windows":
        print("✓ 可以构建Windows版本")
        print("运行: cd app && python build_executable.py")
    elif current_platform == "Darwin":
        print("✓ 可以构建macOS版本") 
        print("运行: cd app && python build_executable.py")
    elif current_platform == "Linux":
        print("✓ 可以构建Linux版本")
        print("运行: cd app && python build_executable.py")
    else:
        print(f"未知平台: {current_platform}")
    
    print("\n要构建其他平台版本，请:")
    print("1. 在对应平台上运行构建脚本")
    print("2. 使用GitHub Actions自动构建")
    print("3. 使用虚拟机或云服务")

if __name__ == "__main__":
    main() 