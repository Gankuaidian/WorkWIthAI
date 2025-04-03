import tkinter as tk
from tkinter import ttk, messagebox
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import networkx as nx
from matplotlib.figure import Figure
from tkinter import scrolledtext
import io
from PIL import Image, ImageTk

class LetterToNumberConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("26 letters to 0-25 numbers")
        self.root.geometry("800x600")
        
        # 主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建界面1: A-Z到0-25的转换
        self.create_converter_interface()
        
        # 模型构建相关变量
        self.model = None
        self.layers = []
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 1
        self.loss_func = "CrossEntropyLoss"  # 默认损失函数
        
        # 训练数据
        self.prepare_training_data()
        
        # 添加窗口关闭事件处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        # 处理窗口关闭事件
        if messagebox.askokcancel("Quit", "Sure to quit?"):
            self.root.destroy()
            exit()

    def prepare_training_data(self):
        # 为26个字母创建简单的特征
        self.X = torch.randn(26, 10)  # 每个字母用10维特征表示
        self.y = torch.tensor([i for i in range(26)])  # 标签为0-25
    
    def create_converter_interface(self):
        # 创建转换界面
        converter_frame = ttk.LabelFrame(self.main_frame, text="Letter to number", padding="10")
        converter_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 输入区域
        input_frame = ttk.Frame(converter_frame)
        input_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(input_frame, text="Input 1 single letter:").pack(side=tk.LEFT, padx=5)
        self.letter_entry = ttk.Entry(input_frame, width=10)
        self.letter_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(input_frame, text="Convert", command=self.convert_letter).pack(side=tk.LEFT, padx=5)
        
        # 结果显示区域
        result_frame = ttk.Frame(converter_frame)
        result_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(result_frame, text="Result:").pack(side=tk.LEFT, padx=5)
        self.result_label = ttk.Label(result_frame, text="")
        self.result_label.pack(side=tk.LEFT, padx=5)
        
        # 按钮区域
        button_frame = ttk.Frame(converter_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Build model", command=self.open_model_builder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Quit", command=self.root.quit).pack(side=tk.RIGHT, padx=5)
    
    def convert_letter(self):
        letter = self.letter_entry.get().upper()
        if len(letter) != 1 or not letter.isalpha():
            messagebox.showerror("Error", "1 single letter is required")
            return
           
        
        # 将字母转换为数字 (A=0, B=1, ..., Z=25)
        number = ord(letter) - ord('A')
        self.result_label.config(text=str(number))
    
    def open_model_builder(self):
        # 创建模型构建窗口
        self.model_window = tk.Toplevel(self.root)
        self.model_window.title("Build Model")
        self.model_window.geometry("900x700")
        
        # 创建模型构建界面
        self.create_model_builder_interface()
    
    def create_model_builder_interface(self):
        # 主框架
        builder_frame = ttk.Frame(self.model_window, padding="10")
        builder_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建一个Notebook (选项卡控件)
        notebook = ttk.Notebook(builder_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 第一个选项卡 - 模型构建
        model_tab = ttk.Frame(notebook)
        notebook.add(model_tab, text="Model Builder")
        
        # 第二个选项卡 - 矩阵变化可视化
        matrix_tab = ttk.Frame(notebook)
        notebook.add(matrix_tab, text="Matrix Visualization")
        
        # 第三个选项卡 - 模型流程图
        flowchart_tab = ttk.Frame(notebook)
        notebook.add(flowchart_tab, text="Model Flowchart")
        
        # 设置模型构建选项卡
        self.setup_model_builder_tab(model_tab)
        
        # 设置矩阵可视化选项卡
        self.setup_matrix_visualization_tab(matrix_tab)
        
        # 设置模型流程图选项卡
        self.setup_flowchart_tab(flowchart_tab)
    
    def setup_model_builder_tab(self, parent):
        # 层添加区域
        layer_frame = ttk.LabelFrame(parent, text="Add layer", padding="10")
        layer_frame.pack(fill=tk.X, pady=10)
        
        # CNN层
        cnn_frame = ttk.Frame(layer_frame)
        cnn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(cnn_frame, text="Add Convolutional layer", command=self.add_cnn_layer).pack(side=tk.LEFT, padx=5)
        ttk.Label(cnn_frame, text="Channels:").pack(side=tk.LEFT, padx=5)
        self.cnn_channels = ttk.Entry(cnn_frame, width=5)
        self.cnn_channels.pack(side=tk.LEFT, padx=5)
        self.cnn_channels.insert(0, "32")
        
        # 线性层
        linear_frame = ttk.Frame(layer_frame)
        linear_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(linear_frame, text="Add Linear layer", command=self.add_linear_layer).pack(side=tk.LEFT, padx=5)
        ttk.Label(linear_frame, text="size:").pack(side=tk.LEFT, padx=5)
        self.linear_size = ttk.Entry(linear_frame, width=5)
        self.linear_size.pack(side=tk.LEFT, padx=5)
        self.linear_size.insert(0, "64")
        
        # Dropout层
        dropout_frame = ttk.Frame(layer_frame)
        dropout_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(dropout_frame, text="Add Dropout layer", command=self.add_dropout_layer).pack(side=tk.LEFT, padx=5)
        ttk.Label(dropout_frame, text="Propability:").pack(side=tk.LEFT, padx=5)
        self.dropout_prob = ttk.Entry(dropout_frame, width=5)
        self.dropout_prob.pack(side=tk.LEFT, padx=5)
        self.dropout_prob.insert(0, "0.5")
        
        # 激活函数
        activation_frame = ttk.Frame(layer_frame)
        activation_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(activation_frame, text="Add activation", command=self.add_activation_layer).pack(side=tk.LEFT, padx=5)
        self.activation_type = tk.StringVar()
        self.activation_type.set("relu")
        ttk.Radiobutton(activation_frame, text="ReLU", variable=self.activation_type, value="relu").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(activation_frame, text="Exp", variable=self.activation_type, value="exp").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(activation_frame, text="Tanh", variable=self.activation_type, value="tanh").pack(side=tk.LEFT, padx=5)
        
        # 池化层
        pool_frame = ttk.Frame(layer_frame)
        pool_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(pool_frame, text="Add Pooling layer", command=self.add_pool_layer).pack(side=tk.LEFT, padx=5)
        ttk.Label(pool_frame, text="size:").pack(side=tk.LEFT, padx=5)
        self.pool_size = ttk.Entry(pool_frame, width=5)
        self.pool_size.pack(side=tk.LEFT, padx=5)
        self.pool_size.insert(0, "1")
        
        # 损失函数选择
        loss_frame = ttk.Frame(layer_frame)
        loss_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(loss_frame, text="Loss function:").pack(side=tk.LEFT, padx=5)
        self.loss_func_var = tk.StringVar()
        self.loss_func_var.set("CrossEntropyLoss")
        loss_combo = ttk.Combobox(loss_frame, textvariable=self.loss_func_var, width=20)
        loss_combo['values'] = ('CrossEntropyLoss', 'MSELoss', 'L1Loss', 'BCELoss', 'NLLLoss')
        loss_combo.pack(side=tk.LEFT, padx=5)
        loss_combo.bind('<<ComboboxSelected>>', self.update_loss_function)
        
        # 批次大小
        batch_frame = ttk.Frame(layer_frame)
        batch_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(batch_frame, text="Batch size:").pack(side=tk.LEFT, padx=5)
        self.batch_size_entry = ttk.Entry(batch_frame, width=5)
        self.batch_size_entry.pack(side=tk.LEFT, padx=5)
        self.batch_size_entry.insert(0, "1")
        
        # 模型层显示区域
        model_display_frame = ttk.LabelFrame(parent, text="Model architect", padding="10")
        model_display_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 使用scrolledtext代替普通text，支持滚动
        self.model_display = scrolledtext.ScrolledText(model_display_frame, height=10, width=50)
        self.model_display.pack(fill=tk.BOTH, expand=True)
        
        # 训练和结果区域
        train_frame = ttk.Frame(parent)
        train_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(train_frame, text="Start Training", command=self.start_training).pack(side=tk.LEFT, padx=5)
        
        # 添加预测随机样本按钮
        ttk.Button(train_frame, text="Random sample predict", command=self.predict_random_sample).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(train_frame, text="Clear Model", command=self.clear_model).pack(side=tk.LEFT, padx=5)
        
        # 预测结果标签
        self.predict_result_label = ttk.Label(train_frame, text="")
        self.predict_result_label.pack(side=tk.LEFT, padx=5)
        
        # 结果显示区域
        self.result_frame = ttk.LabelFrame(parent, text="Result", padding="10")
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 图表区域
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.result_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.result_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # 进度标签
        self.progress_label = ttk.Label(self.result_frame, text="")
        self.progress_label.pack(pady=5)
    
    def predict_random_sample(self):
        if self.model is None:
            messagebox.showinfo("Error", "Please build a model first.")
            return
            return
        
        # 随机选择一个样本
        sample_idx = np.random.randint(0, len(self.X))
        sample_input = self.X[sample_idx:sample_idx+1]
        true_label = self.y[sample_idx].item()
        
        # 使用模型预测
        with torch.no_grad():
            sample_output = self.model(sample_input)
            _, sample_pred = torch.max(sample_output, 1)
            pred_label = sample_pred.item()
        
        # 创建预测结果可视化窗口
        pred_window = tk.Toplevel(self.model_window)
        pred_window.title("Predict sample")
        pred_window.geometry("600x400")
        
        # 创建图表
        fig = Figure(figsize=(8, 6))
        canvas = FigureCanvasTkAgg(fig, master=pred_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建子图布局
        gs = fig.add_gridspec(2, 1)
        
        # 显示样本输入
        ax_input = fig.add_subplot(gs[0])
        ax_input.bar(range(len(sample_input[0])), sample_input[0].detach().numpy())
        ax_input.set_title(f'Sample feature (letter: {chr(65+true_label)})')
        ax_input.set_xlabel('feature')
        ax_input.set_ylabel('value')
        
        # 显示预测结果
        ax_output = fig.add_subplot(gs[1])
        probs = torch.softmax(sample_output, dim=1)[0].detach().numpy()
        top5_indices = np.argsort(probs)[-5:][::-1]
        top5_values = probs[top5_indices]
        top5_labels = [chr(65+i) for i in top5_indices]
        
        ax_output.bar(top5_labels, top5_values)
        ax_output.set_title(f'Prediction: {chr(65+pred_label)} (Real: {chr(65+true_label)})')
        ax_output.set_xlabel('letter')
        ax_output.set_ylabel('probability')
        
        # 更新画布
        fig.tight_layout()
        canvas.draw()
        
        # 在界面2更新预测结果标签
        result_text = f"Prediction: real letter {chr(65+true_label)} → Predicted {chr(65+pred_label)} {'✓' if pred_label == true_label else '✗'}"
        self.predict_result_label.config(text=result_text)
    
    def setup_matrix_visualization_tab(self, parent):
        # 矩阵可视化区域
        self.matrix_canvas_frame = ttk.Frame(parent)
        self.matrix_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 创建一个Figure用于显示矩阵变化
        self.matrix_fig = Figure(figsize=(8, 6))
        self.matrix_canvas = FigureCanvasTkAgg(self.matrix_fig, master=self.matrix_canvas_frame)
        self.matrix_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加说明文本
        ttk.Label(parent, text="Matrix changes Visualization").pack(pady=5)
    
    def setup_flowchart_tab(self, parent):
        # 流程图区域
        self.flowchart_canvas_frame = ttk.Frame(parent)
        self.flowchart_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 创建一个Figure用于显示模型流程图
        self.flowchart_fig = Figure(figsize=(8, 6))
        self.flowchart_canvas = FigureCanvasTkAgg(self.flowchart_fig, master=self.flowchart_canvas_frame)
        self.flowchart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加说明文本
        ttk.Label(parent, text="Flowchart of forward-backward model").pack(pady=5)
    
    def update_loss_function(self, event=None):
        self.loss_func = self.loss_func_var.get()
        if self.loss_func == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_func == "MSELoss":
            self.criterion = nn.MSELoss()
        elif self.loss_func == "L1Loss":
            self.criterion = nn.L1Loss()
        elif self.loss_func == "BCELoss":
            self.criterion = nn.BCELoss()
        elif self.loss_func == "NLLLoss":
            self.criterion = nn.NLLLoss()
        
        # 更新流程图以反映新的损失函数
        if hasattr(self, 'flowchart_fig'):
            self.update_flowchart()
    
    def clear_model(self):
        self.layers = []
        self.model_display.delete(1.0, tk.END)
        
        # 确保flowchart_fig已初始化后再调用update_flowchart
        if hasattr(self, 'flowchart_fig'):
            self.update_flowchart()
    
    def add_cnn_layer(self):
        try:
            channels = int(self.cnn_channels.get())
            if channels <= 0:
                raise ValueError("Positive value only")
            
            layer_info = f"CNN layer (channels: {channels})"
            code_info = f"nn.Conv2d(in_channels=3, out_channels={channels}, kernel_size=3, stride=1, padding=1)"
            self.layers.append(("cnn", channels, code_info))
            self.update_model_display(layer_info, code_info)
            
            # 确保flowchart_fig已初始化后再调用update_flowchart
            if hasattr(self, 'flowchart_fig'):
                self.update_flowchart()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
    
    def add_linear_layer(self):
        try:
            size = int(self.linear_size.get())
            if size <= 0:
                raise ValueError("Positive value only")
            
            # 确定输入大小
            input_size = 10  # 默认输入大小
            if self.layers:
                for layer_type, param, _ in reversed(self.layers):
                    if layer_type == "linear":
                        input_size = param
                        break
            
            layer_info = f"Linear layer (size: {size})"
            code_info = f"nn.Linear(in_features={input_size}, out_features={size})"
            self.layers.append(("linear", size, code_info))
            self.update_model_display(layer_info, code_info)
            
            # 确保flowchart_fig已初始化后再调用update_flowchart
            if hasattr(self, 'flowchart_fig'):
                self.update_flowchart()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
    
    def add_dropout_layer(self):
        try:
            prob = float(self.dropout_prob.get())
            if prob < 0 or prob > 1:
                raise ValueError("Possibility value must in range(0,1)")
            
            layer_info = f"Dropout layer (Probability: {prob})"
            code_info = f"nn.Dropout(p={prob})"
            self.layers.append(("dropout", prob, code_info))
            self.update_model_display(layer_info, code_info)
            
            # 确保flowchart_fig已初始化后再调用update_flowchart
            if hasattr(self, 'flowchart_fig'):
                self.update_flowchart()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
    
    def add_activation_layer(self):
        activation_type = self.activation_type.get()
        layer_info = f"Activation function: {activation_type}"
        
        if activation_type == "relu":
            code_info = "nn.ReLU()"
        elif activation_type == "exp":
            code_info = "lambda x: torch.exp(x)"
        elif activation_type == "tanh":
            code_info = "nn.Tanh()"
        
        self.layers.append(("activation", activation_type, code_info))
        self.update_model_display(layer_info, code_info)
        
        # 确保flowchart_fig已初始化后再调用update_flowchart
        if hasattr(self, 'flowchart_fig'):
            self.update_flowchart()
    
    def add_pool_layer(self):
        try:
            size = int(self.pool_size.get())
            if size <= 0:
                raise ValueError("Positive value only")
            
            layer_info = f"Pooling layer (size: {size})"
            code_info = f"nn.MaxPool2d(kernel_size={size}, stride={size})"
            self.layers.append(("pool", size, code_info))
            self.update_model_display(layer_info, code_info)
            
            # 确保flowchart_fig已初始化后再调用update_flowchart
            if hasattr(self, 'flowchart_fig'):
                self.update_flowchart()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
    
    def update_model_display(self, layer_info, code_info):
        self.model_display.insert(tk.END, f"{layer_info}\ncode: {code_info}\n\n")
    
    def update_flowchart(self):
        # 清除之前的图表
        self.flowchart_fig.clear()
        
        if not self.layers:
            # 如果没有层，仍然显示基本的前向/反向传播循环
            self.draw_basic_cycle()
            self.flowchart_canvas.draw()
            return
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 设置图形布局
        # 前向传播在左侧，反向传播在右侧
        
        # 添加输入节点
        input_shape = f"[{self.batch_size}, 10]"
        G.add_node(f"Input\n{input_shape}", pos=(-5, 0), color='lightblue')
        
        prev_node = f"Input\n{input_shape}"
        y_pos = -1
        
        # 跟踪当前形状
        current_shape = [self.batch_size, 10]
        is_4d = False  # 跟踪当前是否为4D张量
        
        # 添加层节点和边 (前向传播部分)
        layer_nodes = []  # 存储层节点名称，用于后续连接反向传播
        
        for i, (layer_type, param, code) in enumerate(self.layers):
            # 更新形状信息
            if layer_type == "linear":
                if is_4d:  # 如果之前是4D，需要先展平
                    # 添加展平节点
                    flatten_size = current_shape[1] * current_shape[2] * current_shape[3]
                    flatten_shape = [current_shape[0], flatten_size]
                    flatten_node = f"flatten_{i}\n{flatten_shape}"
                    G.add_node(flatten_node, pos=(-5, y_pos), color='lightgreen')
                    G.add_edge(prev_node, flatten_node)
                    prev_node = flatten_node
                    y_pos -= 1
                    layer_nodes.append(flatten_node)
                    
                    # 更新当前形状
                    current_shape = flatten_shape
                    is_4d = False
                
                # 更新线性层后的形状
                current_shape = [current_shape[0], param]
            
            elif layer_type == "cnn":
                if not is_4d:  # 从2D到4D
                    # 计算特征维度
                    feature_dim = int(np.sqrt(current_shape[1]))
                    # 确保是完全平方数
                    if feature_dim * feature_dim != current_shape[1]:
                        feature_dim = int(np.sqrt(current_shape[1])) + 1
                    
                    # 添加重塑节点
                    reshape_shape = [current_shape[0], 1, feature_dim, feature_dim]
                    reshape_node = f"reshape_{i}\n{reshape_shape}"
                    G.add_node(reshape_node, pos=(-5, y_pos), color='lightgreen')
                    G.add_edge(prev_node, reshape_node)
                    prev_node = reshape_node
                    y_pos -= 1
                    layer_nodes.append(reshape_node)
                    
                    # 更新当前形状
                    current_shape = reshape_shape
                    is_4d = True
                
                # 更新CNN层后的形状（保持空间维度不变）
                current_shape = [current_shape[0], param, current_shape[2], current_shape[3]]
            
            elif layer_type == "pool" and is_4d:
                # 更新池化层后的形状
                h, w = current_shape[2], current_shape[3]
                new_h, new_w = h // param, w // param
                # 确保尺寸至少为1
                new_h, new_w = max(1, new_h), max(1, new_w)
                current_shape = [current_shape[0], current_shape[1], new_h, new_w]
            
            # 创建当前层节点
            shape_str = str(current_shape).replace(" ", "")
            node_name = f"{layer_type}_{i}\n{param}\n{shape_str}"
            G.add_node(node_name, pos=(-5, y_pos), color='lightblue')
            G.add_edge(prev_node, node_name)
            prev_node = node_name
            y_pos -= 1
            layer_nodes.append(node_name)
        
        # 添加最终输出层（如果需要）
        if is_4d or (current_shape[1] != 26 and len(current_shape) == 2):
            # 如果是4D，先添加展平节点
            if is_4d:
                flatten_size = current_shape[1] * current_shape[2] * current_shape[3]
                flatten_shape = [current_shape[0], flatten_size]
                flatten_node = f"flatten_final\n{flatten_shape}"
                G.add_node(flatten_node, pos=(-5, y_pos), color='lightgreen')
                G.add_edge(prev_node, flatten_node)
                prev_node = flatten_node
                y_pos -= 1
                layer_nodes.append(flatten_node)
                current_shape = flatten_shape
            
            # 添加最终线性层
            final_shape = [current_shape[0], 26]
            final_node = f"linear_final\n26\n{final_shape}"
            G.add_node(final_node, pos=(-5, y_pos), color='lightblue')
            G.add_edge(prev_node, final_node)
            prev_node = final_node
            y_pos -= 1
            layer_nodes.append(final_node)
            current_shape = final_shape
        
        # 添加输出节点
        output_shape = str(current_shape).replace(" ", "")
        output_node = f"Output\n{output_shape}"
        G.add_node(output_node, pos=(-5, y_pos), color='lightblue')
        G.add_edge(prev_node, output_node)
        layer_nodes.append(output_node)
        
        # 添加损失函数节点
        loss_node = f"Loss\n{self.loss_func}"
        y_pos -= 1
        G.add_node(loss_node, pos=(-5, y_pos), color='salmon')
        G.add_edge(output_node, loss_node)
        
        # 添加目标值节点
        target_node = "Target\n[batch_size]"
        G.add_node(target_node, pos=(0, y_pos), color='lightgreen')
        G.add_edge(target_node, loss_node)
        
        # 反向传播部分 - 从损失函数开始向上传播梯度
        # 反向传播节点位于右侧
        prev_back_node = loss_node
        
        # 反向传播通过所有层（从后向前）
        for i, node in enumerate(reversed(layer_nodes)):
            # 创建梯度节点
            grad_node = f"Grad_{len(layer_nodes)-i-1}"
            G.add_node(grad_node, pos=(5, -len(layer_nodes)+i), color='lightyellow')
            G.add_edge(prev_back_node, grad_node)
            G.add_edge(grad_node, node, style='dashed')  # 虚线表示梯度流
            prev_back_node = grad_node
        
        # 添加优化器节点
        optimizer_node = "Optimizer\nupdate weights"
        G.add_node(optimizer_node, pos=(5, 0), color='lightgreen')
        G.add_edge(prev_back_node, optimizer_node)
        
        # 添加循环边 - 从优化器回到输入，形成训练循环
        G.add_edge(optimizer_node, f"Input\n{input_shape}", style='dashed')
        
        # 绘制流程图
        ax = self.flowchart_fig.add_subplot(111)
        pos = nx.get_node_attributes(G, 'pos')
        
        # 获取节点颜色
        node_colors = [data.get('color', 'lightblue') for _, data in G.nodes(data=True)]
        
        # 获取边样式
        edge_styles = nx.get_edge_attributes(G, 'style')
        solid_edges = [(u, v) for u, v in G.edges() if (u, v) not in edge_styles]
        dashed_edges = [(u, v) for u, v in G.edges() if edge_styles.get((u, v)) == 'dashed']
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=3500, alpha=0.8)
        
        # 绘制实线边
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=solid_edges, 
                              arrowsize=20, width=2, edge_color='black')
        
        # 绘制虚线边
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=dashed_edges, 
                              arrowsize=20, width=1.5, edge_color='red', 
                              style='dashed')
        
        # 添加标签
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold')
        
        # 添加标题
        ax.set_title(f"Model flowchart (Loss func: {self.loss_func})")
        
        # 添加前向/反向传播标签
        ax.text(-5, 1, "Forward propagation", fontsize=12, ha='center', bbox=dict(facecolor='lightblue', alpha=0.5))
        ax.text(5, 1, "Backward propagation", fontsize=12, ha='center', bbox=dict(facecolor='lightyellow', alpha=0.5))
        
        # 关闭坐标轴
        ax.axis('off')
        
        # 更新画布
        self.flowchart_canvas.draw()
    
    def draw_basic_cycle(self):
        """绘制基本的前向/反向传播循环"""
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点
        G.add_node("Input\n[batch_size, 10]", pos=(-5, 0), color='lightblue')
        G.add_node("Model", pos=(-5, -2), color='lightblue')
        G.add_node("Output", pos=(-5, -4), color='lightblue')
        G.add_node(f"Loss\n{self.loss_func}", pos=(-5, -6), color='salmon')
        G.add_node("Target", pos=(0, -6), color='lightgreen')
        G.add_node("Gradients", pos=(5, -4), color='lightyellow')
        G.add_node("Optimizer", pos=(5, -2), color='lightgreen')
        
        # 添加边
        # 前向传播
        G.add_edge("Input\n[batch_size, 10]", "Model")
        G.add_edge("Model", "Output")
        G.add_edge("Output", f"Loss\n{self.loss_func}")
        G.add_edge("Target", f"Loss\n{self.loss_func}")
        
        # 反向传播
        G.add_edge(f"Loss\n{self.loss_func}", "Gradients")
        G.add_edge("Gradients", "Model", style='dashed')
        G.add_edge("Gradients", "Optimizer")
        G.add_edge("Optimizer", "Input\n[batch_size, 10]", style='dashed')
        
        # 绘制流程图
        ax = self.flowchart_fig.add_subplot(111)
        pos = nx.get_node_attributes(G, 'pos')
        
        # 获取节点颜色
        node_colors = [data.get('color', 'lightblue') for _, data in G.nodes(data=True)]
        
        # 获取边样式
        edge_styles = nx.get_edge_attributes(G, 'style')
        solid_edges = [(u, v) for u, v in G.edges() if (u, v) not in edge_styles]
        dashed_edges = [(u, v) for u, v in G.edges() if edge_styles.get((u, v)) == 'dashed']
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=3500, alpha=0.8)
        
        # 绘制实线边
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=solid_edges, 
                              arrowsize=20, width=2, edge_color='black')
        
        # 绘制虚线边
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=dashed_edges, 
                              arrowsize=20, width=1.5, edge_color='red', 
                              style='dashed')
        
        # 添加标签
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
        
        # 添加标题
        ax.set_title(f"Training iteration (Loss func: {self.loss_func})")
        
        # 添加前向/反向传播标签
        ax.text(-5, 1, "Forward propagation", fontsize=12, ha='center', bbox=dict(facecolor='lightblue', alpha=0.5))
        ax.text(5, 1, "Backward propagation", fontsize=12, ha='center', bbox=dict(facecolor='lightyellow', alpha=0.5))
        
        # 关闭坐标轴
        ax.axis('off')
    
    def start_training(self):
        try:
            # 获取批次大小
            self.batch_size = int(self.batch_size_entry.get())
            if self.batch_size <= 0:
                raise ValueError("Positive value only")
            
            # 构建模型
            self.build_model()
            
            # 更新矩阵可视化
            self.update_matrix_visualization()
            
            # 更新流程图
            self.update_flowchart()
            
            # 设置优化器
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
            
            # 训练模型
            epochs = 100
            losses = []
            
            # 重置进度条
            self.progress_var.set(0)
            self.progress_label.config(text="Still training...")
            
            # 开始训练
            for epoch in range(epochs):
                # 前向传播
                outputs = self.model(self.X)
                loss = self.criterion(outputs, self.y)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 记录损失
                losses.append(loss.item())
                
                # 更新进度条
                progress = (epoch + 1) / epochs * 100
                self.progress_var.set(progress)
                self.progress_label.config(text=f"Training process: {epoch+1}/{epochs}, loss: {loss.item():.4f}")
                
                # 更新UI
                self.root.update()
            
            # 显示训练结果
            self.show_training_results(losses)
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Error during training: {str(e)}")
    
    def build_model(self):
        # 改进的模型构建，自动处理层之间的维度转换
        class SimpleModel(nn.Module):
            def __init__(self, layers, batch_size):
                super(SimpleModel, self).__init__()
                self.model_layers = nn.ModuleList()
                self.layer_types = []  # 存储层类型，用于forward中的特殊处理
                
                input_size = 10  # 初始输入特征维度
                is_4d = False    # 跟踪当前是否为4D张量
                feature_h, feature_w = 0, 0  # 初始化特征图尺寸变量
                
                for layer_type, param, _ in layers:
                    if layer_type == "linear":
                        if is_4d:
                            # 添加展平层
                            self.model_layers.append(nn.Flatten())
                            self.layer_types.append("flatten")
                            is_4d = False
                            # 计算展平后的特征数
                            input_size = input_size * feature_h * feature_w
                        
                        self.model_layers.append(nn.Linear(input_size, param))
                        self.layer_types.append("linear")
                        input_size = param
                    
                    elif layer_type == "dropout":
                        self.model_layers.append(nn.Dropout(param))
                        self.layer_types.append("dropout")
                    
                    elif layer_type == "activation":
                        if param == "relu":
                            self.model_layers.append(nn.ReLU())
                        elif param == "exp":
                            # 使用nn.Module子类代替lambda函数
                            class ExpActivation(nn.Module):
                                def forward(self, x):
                                    return torch.exp(x)
                            self.model_layers.append(ExpActivation())
                        elif param == "tanh":
                            self.model_layers.append(nn.Tanh())
                        self.layer_types.append("activation")
                    
                    elif layer_type == "cnn":
                        if not is_4d:
                            # 将2D输入重塑为4D
                            feature_dim = int(np.sqrt(input_size))
                            # 使用nn.Module子类代替lambda函数
                            class Reshape(nn.Module):
                                def __init__(self, feature_dim):
                                    super(Reshape, self).__init__()
                                    self.feature_dim = feature_dim
                                
                                def forward(self, x):
                                    return x.view(x.size(0), 1, self.feature_dim, self.feature_dim)
                            
                            self.model_layers.append(Reshape(feature_dim))
                            self.layer_types.append("reshape")
                            is_4d = True
                            feature_h, feature_w = feature_dim, feature_dim
                            input_size = 1  # 通道数
                        
                        # 添加卷积层
                        self.model_layers.append(nn.Conv2d(input_size, param, kernel_size=3, padding=1))
                        self.layer_types.append("cnn")
                        input_size = param  # 更新通道数
                    
                    elif layer_type == "pool":
                        if is_4d:
                            self.model_layers.append(nn.MaxPool2d(kernel_size=param, stride=param))
                            self.layer_types.append("pool")
                            # 更新特征图尺寸
                            feature_h = feature_h // param
                            feature_w = feature_w // param
                
                # 如果最后一层不是线性层且输出不是26，添加最终线性层
                if is_4d or (input_size != 26 and not is_4d):
                    if is_4d:
                        # 添加展平层
                        self.model_layers.append(nn.Flatten())
                        self.layer_types.append("flatten")
                        # 计算展平后的特征数
                        input_size = input_size * feature_h * feature_w
                    
                    # 添加最终线性层
                    self.model_layers.append(nn.Linear(input_size, 26))
                    self.layer_types.append("linear")
            
            def forward(self, x):
                for layer, layer_type in zip(self.model_layers, self.layer_types):
                    x = layer(x)
                return x
        
        # 创建模型实例
        self.model = SimpleModel(self.layers, self.batch_size)
        
        # 打印模型结构
        print(self.model)
    
    def update_matrix_visualization(self):
        # 清除之前的图表
        self.matrix_fig.clear()
        
        if not self.layers:
            self.matrix_canvas.draw()
            return
        
        # 创建示例输入矩阵
        example_input = torch.randn(self.batch_size, 10)
        
        # 跟踪矩阵在每一层的变化
        matrices = [example_input]
        current_matrix = example_input
        shapes = [f"Input: {tuple(example_input.shape)}"]
        is_4d = False
        
        # 模拟每一层的矩阵变换
        for i, (layer_type, param, _) in enumerate(self.layers):
            if layer_type == "linear":
                if is_4d:  # 如果之前是4D，需要先展平
                    # 正确计算展平后的大小
                    flatten_size = current_matrix.shape[1] * current_matrix.shape[2] * current_matrix.shape[3]
                    current_matrix = current_matrix.view(current_matrix.size(0), -1)
                    shapes.append(f"flattern: {tuple(current_matrix.shape)}")
                    matrices.append(current_matrix.clone())
                    is_4d = False
                
                # 线性层变换
                in_features = current_matrix.shape[1]
                weight = torch.randn(param, in_features)
                bias = torch.randn(param)
                current_matrix = torch.matmul(current_matrix, weight.t()) + bias
            elif layer_type == "activation":
                # 激活函数变换
                if param == "relu":
                    current_matrix = torch.relu(current_matrix)
                elif param == "exp":
                    current_matrix = torch.exp(current_matrix)
                elif param == "tanh":
                    current_matrix = torch.tanh(current_matrix)
            elif layer_type == "dropout":
                # Dropout层 (训练时模拟)
                mask = torch.rand_like(current_matrix) > param
                current_matrix = current_matrix * mask / (1 - param)
            elif layer_type == "cnn":
                # 简化的CNN层模拟
                if not is_4d:  # 如果是第一层CNN
                    # 将2D输入重塑为4D (batch_size, channels, height, width)
                    feature_dim = int(np.sqrt(current_matrix.shape[1]))
                    current_matrix = current_matrix.view(self.batch_size, 1, feature_dim, feature_dim)
                    is_4d = True
                
                # 模拟卷积操作
                in_channels = current_matrix.shape[1]
                out_channels = param
                # 保持空间维度不变（假设padding='same'）
                h, w = current_matrix.shape[2], current_matrix.shape[3]
                current_matrix = torch.randn(self.batch_size, out_channels, h, w)
            elif layer_type == "pool":
                # 池化层模拟
                h, w = current_matrix.shape[2], current_matrix.shape[3]
                new_h, new_w = h // param, w // param
                # 确保尺寸至少为1
                new_h, new_w = max(1, new_h), max(1, new_w)
                current_matrix = torch.randn(current_matrix.shape[0], current_matrix.shape[1], new_h, new_w)
            
            # 添加到矩阵列表
            matrices.append(current_matrix.clone())
            shapes.append(f"{layer_type}_{i}: {tuple(current_matrix.shape)}")
        
        # 添加最终输出层（26个类别）
        if is_4d:
            # 如果最后一层是4D，需要先展平
            flatten_size = current_matrix.shape[1] * current_matrix.shape[2] * current_matrix.shape[3]
            current_matrix = current_matrix.view(current_matrix.size(0), -1)
            
            # 添加展平层到形状列表
            shapes.append(f"flattern: {tuple(current_matrix.shape)}")
            matrices.append(current_matrix.clone())
        
        # 如果最后一层不是26个输出，添加最终的线性层
        if current_matrix.shape[1] != 26:
            in_features = current_matrix.shape[1]
            weight = torch.randn(26, in_features)
            bias = torch.randn(26)
            final_output = torch.matmul(current_matrix, weight.t()) + bias
            
            # 添加最终输出到列表
            shapes.append(f"Final output: {tuple(final_output.shape)}")
            matrices.append(final_output.clone())
        
        # 创建一个子图用于显示尺寸变化
        gs = self.matrix_fig.add_gridspec(2, 1, height_ratios=[1, 3])
        
        # 在顶部添加尺寸变化图
        ax_shapes = self.matrix_fig.add_subplot(gs[0])
        ax_shapes.axis('off')
        shape_text = " → ".join(shapes)
        ax_shapes.text(0.5, 0.5, f"Matrix size changes: {shape_text}", 
                      ha='center', va='center', wrap=True, fontsize=10)
        
        # 确定子图布局
        n_matrices = len(matrices)
        cols = min(3, n_matrices)
        rows = int(np.ceil(n_matrices / cols))
        
        # 为矩阵创建子图区域
        matrix_grid = gs[1].subgridspec(rows, cols)
        
        # 绘制每个矩阵
        for i, matrix in enumerate(matrices):
            ax = self.matrix_fig.add_subplot(matrix_grid[i // cols, i % cols])
            
            # 处理不同维度的张量显示
            if len(matrix.shape) == 2:
                im = ax.imshow(matrix.detach().numpy(), cmap='viridis')
            elif len(matrix.shape) == 4:
                # 对于4D张量，显示第一个样本的第一个通道
                im = ax.imshow(matrix[0, 0].detach().numpy(), cmap='viridis')
            else:
                # 对于其他维度，将其展平为2D
                im = ax.imshow(matrix.reshape(matrix.shape[0], -1).detach().numpy(), cmap='viridis')
            
            self.matrix_fig.colorbar(im, ax=ax)
            
            if i == 0:
                title = f"Input matrix\n{tuple(matrix.shape)}"
            elif i == n_matrices - 1:
                title = f"Final output (26 possible classes)\n{tuple(matrix.shape)}"
            elif i == n_matrices - 2 and len(matrices) > 2 and matrices[-1].shape[1] == 26:
                title = f"Pre-output\n{tuple(matrix.shape)}"
            elif i < len(shapes) and "flattern" in shapes[i]:
                title = f"flattern\n{tuple(matrix.shape)}"
            else:
                layer_idx = i - 1
                if layer_idx < len(self.layers):
                    layer_type, param, _ = self.layers[layer_idx]
                    title = f"{layer_type} layer\n{tuple(matrix.shape)}"
                else:
                    title = f"layer {i}\n{tuple(matrix.shape)}"
            
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("feature", fontsize=8)
            ax.set_ylabel("sample", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
        
        self.matrix_fig.tight_layout()
        self.matrix_canvas.draw()
    
    def show_training_results(self, losses):
        # 清除之前的图表
        self.fig.clear()
        
        # 创建子图布局
        gs = self.fig.add_gridspec(1, 1)  # 修改为只显示损失曲线
        
        # 绘制损失曲线
        ax_loss = self.fig.add_subplot(gs[0, 0])
        ax_loss.plot(losses)
        ax_loss.set_title('Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss value')
        
        # 测试模型
        with torch.no_grad():
            outputs = self.model(self.X)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == self.y).sum().item() / self.y.size(0)
        
        # 更新画布
        self.fig.tight_layout()
        self.canvas.draw()
        
        # 显示准确率
        accuracy_label = ttk.Label(self.result_frame, text=f"Accuracy rate: {accuracy:.2%}")
        accuracy_label.pack(pady=5)
        
        # 更新进度标签
        self.progress_label.config(text=f"Training Finished! Accuracy rate: {accuracy:.2%}")

# 主程序入口
if __name__ == "__main__":
    root = tk.Tk()
    app = LetterToNumberConverter(root)
    root.mainloop()