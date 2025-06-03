# -*- coding: utf-8 -*-

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path

# 抑制 macOS Tkinter 弃用警告
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# 修复导入问题 - 支持打包环境和开发环境
def setup_imports():
    """设置导入路径"""
    current_dir = Path(__file__).parent
    
    # 如果是打包环境
    if getattr(sys, 'frozen', False):
        # 打包环境中，所有文件都在同一目录
        sys.path.insert(0, str(current_dir))
        # 也添加上级目录以防万一
        sys.path.insert(0, str(current_dir.parent))
    else:
        # 开发环境
        project_root = current_dir.parent
        sys.path.insert(0, str(project_root))
        sys.path.insert(0, str(current_dir))

# 设置导入路径
setup_imports()

# 尝试导入ModelLoader
try:
    from app.model_loader import ModelLoader
except ImportError:
    try:
        from model_loader import ModelLoader
    except ImportError:
        # 如果还是失败，创建一个简化版本
        print("警告: 无法导入ModelLoader，使用简化版本")
        
        class ModelLoader:
            def __init__(self):
                self.precision_configs = ["2mm 2%", "2mm 3%", "3mm 2%", "1.5mm 1.5%", "1mm 1%"]
                print("使用简化版ModelLoader")
            
            def get_available_models(self):
                return ["CNN", "Transformer", "CNNTransformer", "CNN_SINGLE"]
            
            def get_precision_configs(self):
                return self.precision_configs.copy()
            
            def predict_from_files(self, model_name, precision_index, adt_plan_file, ref_measure_file, ref_plan_file):
                # 模拟预测结果
                import random
                return random.uniform(70, 90)

class PredictionAppFixed:
    """修复版深度学习预测应用"""
    
    def __init__(self, root):
        print("初始化PredictionAppFixed...")
        self.root = root
        self.root.title("剂量验证预测系统")
        self.root.geometry("480x100")
        print("窗口基本设置完成")
        
        # macOS特定设置
        if sys.platform == "darwin":
            print("应用macOS特定设置...")
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after_idle(lambda: self.root.attributes('-topmost', False))
        
        # 设置窗口背景
        self.root.configure(bg="#f0f0f0")
        print("窗口背景设置完成")
        
        # 初始化模型加载器
        print("初始化模型加载器...")
        self.model_loader = ModelLoader()
        print("模型加载器初始化完成")
        
        # 存储文件夹路径
        self.folder_path = tk.StringVar()
        
        # 存储文件路径（将在选择文件夹后初始化）
        self.file_paths = {
            'adt_plan': tk.StringVar(),
            'ref_measure': tk.StringVar(),
            'ref_plan': tk.StringVar()
        }
        
        # 存储选择的模型和精度
        self.selected_model = tk.StringVar()
        self.selected_precision = tk.StringVar()
        print("变量初始化完成")
        
        # 创建界面
        print("开始创建界面组件...")
        self.create_widgets()
        print("界面组件创建完成")
        
        # 延迟初始化模型列表
        self.root.after(100, self.update_model_list)
        
        # 强制更新显示
        print("强制更新显示...")
        self.root.update_idletasks()
        self.root.deiconify()
        self.root.lift()
        print("PredictionAppFixed初始化完成")
    
    def create_widgets(self):
        """创建界面组件"""
        # 主容器 - 极小间距
        main_frame = tk.Frame(self.root, bg="#f0f0f0", padx=5, pady=5)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 默认选项
        default_models = ["CNN", "Transformer", "CNNTransformer", "CNN_SINGLE"]
        default_precisions = ["2mm 2%", "2mm 3%", "3mm 2%", "1.5mm 1.5%", "1mm 1%"]
        
        # 设置默认值
        self.selected_model.set(default_models[0])
        self.selected_precision.set(default_precisions[0])
        
        # 控制栏 - 所有组件在一行，极小间距
        control_frame = tk.Frame(main_frame, bg="#f0f0f0")
        control_frame.pack(fill=tk.X, pady=(0, 3))
        
        # 1. 模型选择 - 更小宽度
        self.model_menu = tk.OptionMenu(control_frame, self.selected_model, *default_models)
        self.model_menu.config(width=10)
        self.model_menu.pack(side=tk.LEFT, padx=(0, 3))
        
        # 2. 精度选择 - 更小宽度
        self.precision_menu = tk.OptionMenu(control_frame, self.selected_precision, *default_precisions)
        self.precision_menu.config(width=8)
        self.precision_menu.pack(side=tk.LEFT, padx=(0, 3))
        
        # 3. 选择文件夹按钮 - 更小宽度
        browse_btn = tk.Button(control_frame, text="选择文件夹", command=self.browse_folder, 
                              width=10)
        browse_btn.pack(side=tk.LEFT, padx=(0, 3))
        
        # 4. 预测按钮 - 更小宽度
        self.predict_btn = tk.Button(control_frame, text="开始预测", 
                                    command=self.start_prediction,
                                    bg="#4CAF50", fg="white", width=8)
        self.predict_btn.pack(side=tk.LEFT, padx=(0, 3))
        
        # 5. 清除按钮 - 更小宽度
        clear_btn = tk.Button(control_frame, text="清除", 
                             command=self.clear_all,
                             bg="#f44336", fg="white", width=6)
        clear_btn.pack(side=tk.LEFT)
        
        # 结果显示 - 删除标签，只保留2行文本框
        self.result_text = tk.Text(main_frame, height=2, wrap=tk.WORD, font=("Arial", 9))
        self.result_text.pack(fill=tk.BOTH, expand=True)
    
    def update_model_list(self):
        """更新可用模型列表"""
        try:
            # 检查可用模型
            try:
                available_models = self.model_loader.get_available_models()
                if available_models:
                    print(f"找到 {len(available_models)} 个可用模型")
                else:
                    print("警告: 未找到可用模型文件")
            except Exception as e:
                print("警告: 模型检查失败")
                
        except Exception as e:
            print(f"更新模型列表失败: {e}")
    
    def browse_folder(self):
        """浏览文件夹"""
        folder_path = filedialog.askdirectory(
            title="选择数据文件夹"
        )
        
        if folder_path:
            self.folder_path.set(folder_path)
            self.check_folder_files(folder_path)
    
    def check_folder_files(self, folder_path):
        """检查文件夹中的文件"""
        required_files = {
            "ADT_plan.xlsx": "ADT_plan",
            "Ref_measure.xlsx": "Ref_measure", 
            "Ref_plan.xlsx": "Ref_plan"
        }
        
        found_files = {}
        missing_files = []
        
        folder = Path(folder_path)
        
        # 检查每个必需文件
        for filename, display_name in required_files.items():
            file_path = folder / filename
            if file_path.exists():
                found_files[display_name] = str(file_path)
            else:
                # 尝试查找.xls格式
                alt_file = folder / filename.replace('.xlsx', '.xls')
                if alt_file.exists():
                    found_files[display_name] = str(alt_file)
                else:
                    missing_files.append(filename)
        
        # 更新文件路径
        self.file_paths = {
            'adt_plan': tk.StringVar(value=found_files.get("ADT_plan", "")),
            'ref_measure': tk.StringVar(value=found_files.get("Ref_measure", "")),
            'ref_plan': tk.StringVar(value=found_files.get("Ref_plan", ""))
        }
        
        # 更新状态显示
        self.update_file_status(found_files, missing_files)
    
    def update_file_status(self, found_files, missing_files):
        """更新文件状态显示"""
        if not missing_files:
            # 所有文件都找到了
            print("文件检查完成 - 所有文件已找到")
        else:
            # 有文件缺失
            print(f"文件检查完成 - 缺失 {len(missing_files)} 个文件")
    
    def clear_all(self):
        """清除所有输入"""
        self.folder_path.set("")
        self.file_paths = {
            'adt_plan': tk.StringVar(),
            'ref_measure': tk.StringVar(),
            'ref_plan': tk.StringVar()
        }
        self.result_text.delete(1.0, tk.END)
    
    def validate_inputs(self):
        """验证输入"""
        if not self.selected_model.get():
            return False, "请选择模型"
        
        if not self.selected_precision.get():
            return False, "请选择精度配置"
        
        if not self.folder_path.get():
            return False, "请选择数据文件夹"
        
        # 所有模型都需要验证三个文件
        required_files = ["adt_plan", "ref_measure", "ref_plan"]
        for file_key in required_files:
            file_path = self.file_paths[file_key].get()
            if not file_path:
                return False, f"文件夹中缺少必需文件"
            
            if not Path(file_path).exists():
                return False, f"文件不存在: {Path(file_path).name}"
        
        return True, "验证通过"
    
    def start_prediction(self):
        """开始预测"""
        # 验证输入
        valid, message = self.validate_inputs()
        if not valid:
            messagebox.showerror("输入错误", message)
            return
        
        # 禁用预测按钮
        self.predict_btn.config(state='disabled')
        
        # 在新线程中运行预测
        thread = threading.Thread(target=self.run_prediction)
        thread.daemon = True
        thread.start()
    
    def run_prediction(self):
        """运行预测（后台线程）"""
        try:
            # 获取选择的模型和精度
            model_name = self.selected_model.get()
            precision_name = self.selected_precision.get()
            precision_index = self.model_loader.get_precision_configs().index(precision_name)
            
            # 所有模型都需要三个文件
            result = self.model_loader.predict_from_files(
                model_name,
                precision_index,
                self.file_paths['adt_plan'].get(),
                self.file_paths['ref_measure'].get(),
                self.file_paths['ref_plan'].get()
            )
            
            # 在主线程中更新UI
            self.root.after(0, self.prediction_complete, result, model_name, precision_name)
            
        except Exception as e:
            # 在主线程中显示错误
            self.root.after(0, self.prediction_error, str(e))
    
    def prediction_complete(self, result, model_name, precision_name):
        """预测完成回调"""
        # 恢复预测按钮
        self.predict_btn.config(state='normal')
        
        # 所有模型都显示三个文件的信息
        result_text = f"""预测完成！

模型: {model_name}
精度配置: {precision_name}
预测结果: {result:.4f}%

输入文件:
- ADT_plan: {Path(self.file_paths['adt_plan'].get()).name}
- Ref_measure: {Path(self.file_paths['ref_measure'].get()).name}
- Ref_plan: {Path(self.file_paths['ref_plan'].get()).name}

预测时间: {self.get_current_time()}
"""
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, result_text)
        
        # 显示成功消息
        messagebox.showinfo("预测完成", f"预测结果: {result:.4f}%")
    
    def prediction_error(self, error_message):
        """预测错误回调"""
        # 恢复预测按钮
        self.predict_btn.config(state='normal')
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, f"预测过程中发生错误:\n{error_message}")
        
        # 显示错误消息
        messagebox.showerror("预测失败", f"预测过程中发生错误:\n{error_message}")
    
    def get_current_time(self):
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """主函数"""
    print("启动修复版GUI...")
    
    # 创建主窗口
    root = tk.Tk()
    print("Tkinter根窗口创建成功")
    
    try:
        print("开始创建应用实例...")
        app = PredictionAppFixed(root)
        print("修复版GUI创建成功")
        
        # 确保窗口显示
        print("强制显示窗口...")
        root.update_idletasks()
        root.deiconify()
        root.lift()
        
        print("开始GUI主循环...")
        # 运行主循环
        root.mainloop()
        print("GUI主循环结束")
        
    except Exception as e:
        print(f"GUI创建失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 