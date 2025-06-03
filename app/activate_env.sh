#!/bin/bash
echo "激活虚拟环境..."
source "/Users/neuo/Documents/dv/app/venv/bin/activate"

# 抑制 macOS Tkinter 弃用警告
export TK_SILENCE_DEPRECATION=1

echo "虚拟环境已激活"
echo "当前Python路径: $VIRTUAL_ENV"
echo ""
echo "可用命令:"
echo "  python app/main.py    - 启动应用"
echo "  pip list             - 查看已安装包"
echo "  deactivate           - 退出虚拟环境"
echo ""
exec bash 