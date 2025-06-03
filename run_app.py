#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
剂量验证预测系统 - 快速启动脚本

这个脚本提供了一个简单的方式来启动应用
"""

import os
import sys
from pathlib import Path

def main():
    """主函数"""
    # 获取项目根目录
    project_root = Path(__file__).parent
    
    # 添加app目录到Python路径
    app_dir = project_root / "app"
    sys.path.insert(0, str(app_dir))
    
    # 切换到app目录
    os.chdir(app_dir)
    
    try:
        # 导入并运行主应用
        from main import main as app_main
        app_main()
    except ImportError as e:
        print(f"导入失败: {e}")
        print("请确保app/main.py文件存在")
        sys.exit(1)
    except Exception as e:
        print(f"运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 