#!/bin/bash
echo "在虚拟环境中运行应用..."
source "/Users/neuo/Documents/dv/app/venv/bin/activate"

# 抑制 macOS Tkinter 弃用警告
export TK_SILENCE_DEPRECATION=1

cd "/Users/neuo/Documents/dv"
python app/gui_fixed.py 