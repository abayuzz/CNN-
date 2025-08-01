"""
深层CNN实时手势识别系统 - 详细注释版
作者: GitHub Copilot
日期: 2025年7月31日
功能: 使用训练好的深层CNN模型进行实时手势识别

主要库函数说明:
- cv2: OpenCV计算机视觉库，用于摄像头操作和图像处理
- torch: PyTorch深度学习框架，用于模型加载和推理
- numpy: 数值计算库，用于数组操作和数学计算
- PIL: Python图像库，用于图像格式转换
- torchvision.transforms: 图像预处理工具
- collections.Counter: 用于统计和计数操作
"""

import cv2  # OpenCV - 计算机视觉库
import torch  # PyTorch - 深度学习框架
import torch.nn as nn  # 神经网络模块
import numpy as np  # NumPy - 数值计算库
from torchvision import transforms  # 图像预处理工具
from PIL import Image  # Python图像库
import time  # 时间操作库
import os  # 操作系统接口
from collections import Counter  # 计数器工具

# ================================
# 1. CNN模型定义 (与训练时完全相同)
# ================================

class CNN(nn.Module):
    """
    深层卷积神经网络模型类
    必须与训练时的模型结构完全一致，否则无法加载权重
    
    网络结构: 4层卷积 + 4层全连接
    输入: (3, 224, 224) RGB图像
    输出: (3,) 三个类别的概率分数
    """
    
    def __init__(self):
        """
        初始化网络层
        每一层的参数必须与训练时保持一致
        """
        super(CNN, self).__init__()
        
        # ================================
        # 卷积特征提取部分
        # ================================
        
        # 第一个卷积块 - 基础特征提取
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 3→32通道
        self.bn1 = nn.BatchNorm2d(32)  # 批量归一化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化 224→112
        self.dropout1 = nn.Dropout2d(0.25)  # Dropout正则化
        
        # 第二个卷积块 - 中层特征提取
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 32→64通道
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112→56
        self.dropout2 = nn.Dropout2d(0.25)
        
        # 第三个卷积块 - 高层特征提取
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 64→128通道
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56→28
        self.dropout3 = nn.Dropout2d(0.25)
        
        # 第四个卷积块 - 语义特征提取
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # 128→256通道
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28→14
        self.dropout4 = nn.Dropout2d(0.3)
        
        # ================================
        # 全连接分类部分
        # ================================
        
        # 计算展平后的特征数量: 256 * 14 * 14 = 50176
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # 第一个全连接层
        self.bn_fc1 = nn.BatchNorm1d(512)  # 一维批量归一化
        self.dropout5 = nn.Dropout(0.5)  # 50% Dropout
        
        self.fc2 = nn.Linear(512, 256)  # 第二个全连接层
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout6 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 128)  # 第三个全连接层
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.dropout7 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(128, 3)  # 输出层: 石头、剪刀、布 (3类)

    def forward(self, x):
        """
        前向传播函数
        定义数据在网络中的流动路径
        
        参数:
            x: 输入张量 (batch_size, 3, 224, 224)
            
        返回:
            输出张量 (batch_size, 3) - 三个类别的原始分数
        """
        
        # ================================
        # 卷积特征提取阶段
        # ================================
        
        # 第一个卷积块: (3,224,224) → (32,112,112)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # 第二个卷积块: (32,112,112) → (64,56,56)
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        # 第三个卷积块: (64,56,56) → (128,28,28)
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        # 第四个卷积块: (128,28,28) → (256,14,14)
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)
        
        # ================================
        # 全连接分类阶段
        # ================================
        
        # 展平操作: (256,14,14) → (50176,)
        # view(-1, 256*14*14): -1表示自动计算batch维度
        x = x.view(-1, 256 * 14 * 14)
        
        # 第一个全连接层: (50176,) → (512,)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout5(x)
        
        # 第二个全连接层: (512,) → (256,)
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout6(x)
        
        # 第三个全连接层: (256,) → (128,)
        x = torch.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout7(x)
        
        # 输出层: (128,) → (3,)
        # 不使用激活函数，输出原始分数
        x = self.fc4(x)
        
        return x

# ================================
# 2. 深层手势检测器类
# ================================

class DeepGestureDetector:
    """
    深层CNN实时手势检测器
    
    主要功能:
    1. 加载训练好的CNN模型
    2. 初始化摄像头和预处理管道
    3. 实时捕获和处理视频帧
    4. 使用时间平滑算法提高检测稳定性
    5. 绘制用户界面和检测结果
    """
    
    def __init__(self, model_path='cnn_model.pth'):
        """
        初始化检测器
        
        参数:
            model_path: 模型文件路径，默认为'cnn_model.pth'
        """
        print("初始化深层CNN手势识别系统...")
        
        # ================================
        # 类别标签配置
        # ================================
        
        # datasets.ImageFolder按字母顺序自动排序类别
        # 这个顺序必须与训练时保持一致
        self.class_names = ['paper', 'rock', 'scissors']  # 英文标签
        self.class_names_cn = ['paper', 'rock', 'scissors']  # 中文显示标签
        
        # ================================
        # 时间平滑算法参数
        # ================================
        
        # 预测历史记录，用于时间平滑算法
        self.prediction_history = []  # 存储历史预测结果
        self.history_size = 7  # 最大历史记录长度
        self.confidence_threshold = 0.75  # 置信度阈值，高于此值才显示结果
        
        # ================================
        # 模型初始化和加载
        # ================================
        
        # 创建模型实例
        self.model = CNN()
        
        # torch.device(): 选择计算设备(GPU优先，CPU备用)
        # torch.cuda.is_available(): 检查CUDA是否可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # model.to(device): 将模型移动到指定设备
        self.model.to(self.device)
        
        try:
            # 优先加载最佳模型文件
            if os.path.exists('best_cnn_model.pth'):
                # torch.load(): 加载保存的模型权重
                # map_location: 指定加载到的设备
                self.model.load_state_dict(torch.load('best_cnn_model.pth', map_location=self.device))
                print("✓ 最佳模型加载成功！")
            else:
                # 加载普通模型文件
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("✓ 模型加载成功！")
            
            # model.eval(): 设置模型为评估模式
            # 禁用Dropout和BatchNorm的训练行为
            self.model.eval()
            print(f"✓ 使用设备: {self.device}")
            
        except FileNotFoundError:
            print(f"❌ 错误：找不到模型文件 {model_path}")
            print("请先运行 cnn.py 训练并保存模型")
            return
        except Exception as e:
            print(f"❌ 模型加载错误: {e}")
            return
        
        # ================================
        # 图像预处理管道配置
        # ================================
        
        # transforms.Compose(): 组合多个图像变换
        # 必须与训练时的预处理保持完全一致
        self.transform = transforms.Compose([
            # transforms.Resize(): 调整图像尺寸到224×224
            transforms.Resize((224, 224)),
            
            # transforms.ToTensor(): PIL图像→PyTorch张量
            # 同时将像素值从[0,255]归一化到[0,1]
            # 维度从(H,W,C)转换为(C,H,W)
            transforms.ToTensor(),
            
            # transforms.Normalize(): 标准化处理
            # 使用ImageNet预训练模型的均值和标准差
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # RGB三通道均值
                std=[0.229, 0.224, 0.225]    # RGB三通道标准差
            )
        ])
        
        # ================================
        # 摄像头初始化
        # ================================
        
        # cv2.VideoCapture(): 创建视频捕获对象
        # 参数0表示默认摄像头(通常是笔记本内置摄像头)
        self.cap = cv2.VideoCapture(0)
        
        # cap.isOpened(): 检查摄像头是否成功打开
        if not self.cap.isOpened():
            print("❌ 错误：无法打开摄像头")
            return
        
        # ================================
        # 摄像头参数配置
        # ================================
        
        # cap.set(): 设置摄像头属性
        # CAP_PROP_FRAME_WIDTH: 设置帧宽度为800像素
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        # CAP_PROP_FRAME_HEIGHT: 设置帧高度为600像素
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        # CAP_PROP_FPS: 设置帧率为30fps
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ 摄像头初始化成功！")
        print("\n使用说明：")
        print("- 将手势放在绿色检测框内")
        print("- 保持手势稳定1-2秒获得最佳识别效果")
        print("- 按 'q' 退出程序")
        print("- 按 's' 保存当前帧")
    
    def preprocess_frame(self, frame):
        """
        预处理视频帧，准备输入到CNN模型
        
        参数:
            frame: OpenCV格式的图像帧 (numpy数组)
            
        返回:
            tensor_image: 预处理后的PyTorch张量
        """
        
        # ================================
        # 图像降噪处理
        # ================================
        
        # cv2.GaussianBlur(): 高斯模糊滤波
        # (3,3): 滤波核大小为3×3
        # 0: 自动计算标准差
        # 作用: 减少图像噪声，提高识别稳定性
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # ================================
        # 颜色空间转换
        # ================================
        
        # cv2.cvtColor(): 颜色空间转换
        # COLOR_BGR2RGB: OpenCV默认BGR格式→PIL需要的RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Image.fromarray(): numpy数组→PIL图像对象
        pil_image = Image.fromarray(rgb_frame)
        
        # ================================
        # 应用预处理变换
        # ================================
        
        # 应用之前定义的变换管道
        # 包括: 尺寸调整→张量转换→标准化
        tensor_image = self.transform(pil_image)
        
        # unsqueeze(0): 添加batch维度
        # (3, 224, 224) → (1, 3, 224, 224)
        # 因为模型期望batch输入
        tensor_image = tensor_image.unsqueeze(0)
        
        # to(device): 移动到计算设备(GPU或CPU)
        tensor_image = tensor_image.to(self.device)
        
        return tensor_image
    
    def predict_with_smoothing(self, frame):
        """
        带时间平滑的手势预测函数
        
        核心算法: 使用加权平均对多帧预测结果进行平滑
        目的: 减少单帧误识别，提高检测稳定性
        
        参数:
            frame: 待预测的图像帧
            
        返回:
            tuple: (预测类别, 置信度, 概率分布)
        """
        
        # ================================
        # 单帧预测
        # ================================
        
        # 预处理图像
        input_tensor = self.preprocess_frame(frame)
        
        # 模型推理
        # torch.no_grad(): 禁用梯度计算，节省内存和计算
        with torch.no_grad():
            # 前向传播，获取原始输出分数
            outputs = self.model(input_tensor)  # (1, 3)
            
            # torch.softmax(): 将原始分数转换为概率分布
            # dim=1: 在类别维度上进行softmax
            probabilities = torch.softmax(outputs, dim=1).cpu()  # 移到CPU
            
            # torch.max(): 获取最大概率值和对应索引
            # 返回: (最大值张量, 索引张量)
            confidence, predicted = torch.max(probabilities, 1)
            
            # 提取具体数值
            predicted_class = self.class_names[predicted.item()]  # 预测类别名称
            confidence_score = confidence.item()  # 置信度分数
            
            # ================================
            # 更新历史记录
            # ================================
            
            # 将当前预测添加到历史记录
            # 存储: (类别名, 置信度, 概率数组)
            self.prediction_history.append((
                predicted_class, 
                confidence_score, 
                probabilities[0].numpy()  # 转换为numpy数组
            ))
            
            # 维护历史记录长度
            if len(self.prediction_history) > self.history_size:
                # list.pop(0): 移除最早的记录
                self.prediction_history.pop(0)
            
            # ================================
            # 时间平滑算法
            # ================================
            
            # 只有积累足够历史记录时才进行平滑
            if len(self.prediction_history) >= 3:
                
                # 根据历史记录长度动态生成权重
                history_len = len(self.prediction_history)
                
                # 权重分配策略: 最新帧权重最高，越早权重越低
                if history_len == 3:
                    weights = np.array([0.2, 0.3, 0.5])  # 最新帧权重50%
                elif history_len == 4:
                    weights = np.array([0.1, 0.2, 0.3, 0.4])
                elif history_len == 5:
                    weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
                elif history_len == 6:
                    weights = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.25])
                else:  # history_len >= 7
                    weights = np.array([0.05, 0.1, 0.1, 0.15, 0.2, 0.2, 0.2])
                
                # 确保权重数组长度与历史记录匹配
                weights = weights[:history_len]
                
                # 权重归一化，确保总和为1
                weights = weights / weights.sum()
                
                # ================================
                # 加权平均计算
                # ================================
                
                # 提取所有历史记录的概率分布
                # [item[2] for item in self.prediction_history]: 提取概率数组
                prob_arrays = [item[2] for item in self.prediction_history]
                
                # np.average(): 计算加权平均
                # axis=0: 在样本维度上平均，保持类别维度
                avg_probs = np.average(prob_arrays, weights=weights, axis=0)
                
                # ================================
                # 平滑后的预测结果
                # ================================
                
                # np.argmax(): 找到概率最高的类别索引
                smooth_predicted = np.argmax(avg_probs)
                smooth_confidence = avg_probs[smooth_predicted]  # 对应的置信度
                smooth_class = self.class_names[smooth_predicted]  # 类别名称
                
                return smooth_class, smooth_confidence, avg_probs
            
        # 历史记录不足时，返回单帧预测结果
        return predicted_class, confidence_score, probabilities[0].numpy()
    
    def draw_detection_area(self, frame):
        """
        在视频帧上绘制手势检测区域
        
        参数:
            frame: 视频帧 (numpy数组)
            
        返回:
            detection_roi: 检测区域的图像片段
        """
        
        # 获取视频帧尺寸
        height, width = frame.shape[:2]
        
        # ================================
        # 定义检测区域坐标
        # ================================
        
        # 右上角区域，尺寸330×330像素
        x1, y1 = width - 380, 50  # 左上角坐标
        x2, y2 = width - 50, 380  # 右下角坐标
        
        # ================================
        # 绘制检测框
        # ================================
        
        # cv2.rectangle(): 绘制矩形
        # 外层边框 (深绿色，较粗)
        cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 150, 0), 3)
        # 内层边框 (亮绿色，较细)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # ================================
        # 添加标题文字
        # ================================
        
        # cv2.putText(): 在图像上绘制文字
        # 参数: (图像, 文字, 位置, 字体, 大小, 颜色, 粗细)
        cv2.putText(frame, "Hand Gesture Area", (x1, y1-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ================================
        # 提取检测区域
        # ================================
        
        # frame[y1:y2, x1:x2]: 数组切片，提取指定区域
        # 返回检测区域的图像片段，用于后续的手势识别
        return frame[y1:y2, x1:x2]
    
    def draw_ui(self, frame, gesture, confidence, probabilities):
        """
        绘制用户界面，显示检测结果和相关信息
        
        参数:
            frame: 视频帧
            gesture: 预测的手势类别
            confidence: 预测置信度
            probabilities: 所有类别的概率分布
        """
        
        # ================================
        # 绘制半透明背景面板
        # ================================
        
        # frame.copy(): 复制原始帧，用于混合
        overlay = frame.copy()
        
        # 绘制黑色矩形作为背景
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        
        # cv2.addWeighted(): 图像混合，创建半透明效果
        # 参数: (图像1, 权重1, 图像2, 权重2, 常数项, 输出)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # ================================
        # 根据置信度确定显示内容
        # ================================
        
        if confidence > self.confidence_threshold:
            # 置信度足够高，显示识别结果
            gesture_cn = self.class_names_cn[self.class_names.index(gesture)]
            result_text = f"{gesture_cn} ({gesture})"
            color = (0, 255, 0)  # 绿色表示成功
            status = "识别成功"
        else:
            # 置信度不足，显示未识别状态
            result_text = "未识别"
            color = (0, 165, 255)  # 橙色表示置信度不足
            status = "置信度不足"
        
        # ================================
        # 绘制主要结果信息
        # ================================
        
        # 主要结果文字 (大字体)
        cv2.putText(frame, result_text, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # 置信度信息
        # f"{confidence:.1%}": 格式化为百分比，保留1位小数
        cv2.putText(frame, f"置信度: {confidence:.1%}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 状态信息
        cv2.putText(frame, f"状态: {status}", (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ================================
        # 绘制概率分布条形图
        # ================================
        
        y_offset = 135  # 起始Y坐标
        
        # enumerate(): 同时获取索引和值
        for i, (class_name, class_cn) in enumerate(zip(self.class_names, self.class_names_cn)):
            prob = probabilities[i]  # 当前类别的概率
            
            # 计算概率条的宽度 (最大200像素)
            bar_width = int(prob * 200)
            
            # ================================
            # 绘制概率条
            # ================================
            
            # 填充的概率条
            # 最高概率用绿色，其他用灰色
            bar_color = (0, 255, 0) if i == np.argmax(probabilities) else (100, 100, 100)
            cv2.rectangle(frame, (20, y_offset + i*20), (20 + bar_width, y_offset + i*20 + 15), 
                         bar_color, -1)  # -1表示填充
            
            # 概率条边框
            cv2.rectangle(frame, (20, y_offset + i*20), (220, y_offset + i*20 + 15), 
                         (255, 255, 255), 1)  # 白色边框
            
            # ================================
            # 绘制概率文字
            # ================================
            
            # 格式化概率文字
            prob_text = f"{class_cn}: {prob:.1%}"
            cv2.putText(frame, prob_text, (230, y_offset + i*20 + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # ================================
        # 绘制帮助信息和FPS
        # ================================
        
        # 帮助信息位置 (屏幕底部)
        help_y = frame.shape[0] - 60
        
        # 操作说明
        cv2.putText(frame, "按键: 'q'-退出, 's'-保存截图", (20, help_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # FPS显示
        # getattr(self, 'fps', 0): 获取fps属性，默认值为0
        cv2.putText(frame, f"FPS: {getattr(self, 'fps', 0):.1f}", (20, help_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self):
        """
        主运行循环 - 实时手势识别的核心函数
        
        功能流程:
        1. 连续捕获摄像头画面
        2. 预处理每一帧图像
        3. 使用CNN模型进行预测
        4. 应用时间平滑算法
        5. 绘制检测结果和用户界面
        6. 处理用户交互 (按键响应)
        7. 计算和显示FPS
        """
        
        # ================================
        # 初始状态检查
        # ================================
        
        # hasattr(): 检查对象是否有指定属性
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            print("❌ 摄像头未正确初始化")
            return
        
        print("\n🚀 开始实时手势识别...")
        
        # ================================
        # FPS计算相关变量
        # ================================
        
        frame_count = 0  # 帧计数器
        start_time = time.time()  # 起始时间
        
        # ================================
        # 主检测循环
        # ================================
        
        try:
            while True:  # 无限循环，直到用户退出
                
                # ================================
                # 捕获摄像头画面
                # ================================
                
                # cap.read(): 读取一帧图像
                # 返回: (是否成功, 图像数据)
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ 无法读取摄像头画面")
                    break
                
                # ================================
                # 图像预处理
                # ================================
                
                # cv2.flip(): 水平翻转图像
                # 参数1表示水平翻转，创建镜像效果
                # 让用户感觉更自然，就像照镜子
                frame = cv2.flip(frame, 1)
                
                # 绘制检测区域并获取ROI (Region of Interest)
                detection_roi = self.draw_detection_area(frame)
                
                # ================================
                # 手势识别处理
                # ================================
                
                try:
                    # 使用时间平滑算法进行手势预测
                    gesture, confidence, probabilities = self.predict_with_smoothing(detection_roi)
                    
                    # 绘制用户界面和检测结果
                    self.draw_ui(frame, gesture, confidence, probabilities)
                    
                except Exception as e:
                    # 异常处理: 显示错误信息
                    error_msg = f"检测错误: {str(e)[:30]}"
                    print(f"❌ {error_msg}")  # 控制台输出
                    
                    # 在视频帧上显示错误信息
                    cv2.putText(frame, error_msg, (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 显示解决建议
                    cv2.putText(frame, "请检查模型文件是否正确", (20, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # ================================
                # FPS计算
                # ================================
                
                frame_count += 1
                
                # 每30帧计算一次FPS
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time  # 经过的时间
                    # FPS = 帧数 / 时间
                    self.fps = 30 / elapsed if elapsed > 0 else 0
                    start_time = time.time()  # 重置计时器
                
                # ================================
                # 显示处理后的画面
                # ================================
                
                # cv2.imshow(): 显示图像窗口
                cv2.imshow('深层CNN手势识别系统', frame)
                
                # ================================
                # 用户交互处理
                # ================================
                
                # cv2.waitKey(): 等待按键输入
                # 参数1: 等待1毫秒
                # & 0xFF: 只保留低8位，兼容不同系统
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # 'q' 键: 退出程序
                    print("👋 用户退出程序")
                    break
                    
                elif key == ord('s'):
                    # 's' 键: 保存当前帧截图
                    
                    # time.strftime(): 格式化当前时间
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"gesture_capture_{timestamp}.jpg"
                    
                    # cv2.imwrite(): 保存图像文件
                    cv2.imwrite(filename, frame)
                    print(f"📸 截图已保存: {filename}")
        
        except KeyboardInterrupt:
            # Ctrl+C 中断处理
            print("\n👋 程序被用户中断")
        
        finally:
            # ================================
            # 资源清理
            # ================================
            
            # 无论如何退出，都要清理资源
            
            # cap.release(): 释放摄像头资源
            self.cap.release()
            
            # cv2.destroyAllWindows(): 关闭所有OpenCV窗口
            cv2.destroyAllWindows()
            
            print("✓ 资源清理完成")

# ================================
# 3. 主函数 - 程序入口点
# ================================

def main():
    """
    主函数: 程序的入口点
    负责创建检测器实例并启动检测流程
    """
    
    # 打印程序标题和信息
    print("=" * 60)
    print("深层CNN实时手势识别系统 v2.0")
    print("支持：石头、剪刀、布手势识别")
    print("=" * 60)
    
    # ================================
    # 创建检测器实例
    # ================================
    
    # 实例化深层手势检测器
    detector = DeepGestureDetector()
    
    # ================================
    # 启动检测流程
    # ================================
    
    # 检查检测器是否初始化成功
    if hasattr(detector, 'cap') and detector.cap.isOpened():
        # 启动主检测循环
        detector.run()
    else:
        print("❌ 初始化失败，程序退出")

# ================================
# 程序入口点
# ================================

if __name__ == "__main__":
    """
    当脚本被直接运行时执行main函数
    如果被作为模块导入则不执行
    """
    main()

"""
代码总结:

🏗️ 系统架构:
1. CNN模型类: 与训练时完全相同的深层网络结构
2. 检测器类: 封装所有检测逻辑和UI交互
3. 主函数: 程序入口和流程控制

🔍 核心算法:
1. 图像预处理: 降噪→颜色转换→标准化→张量化
2. 模型推理: CNN前向传播→Softmax概率化
3. 时间平滑: 加权平均多帧预测→提高稳定性
4. 结果显示: 概率条形图→置信度判断→UI绘制

⚙️ 技术亮点:
1. 设备自适应: GPU优先，CPU备用
2. 异常处理: 完善的错误捕获和显示
3. 用户交互: 实时FPS显示，截图保存功能
4. 性能优化: ROI提取，时间平滑算法

🎯 应用价值:
这是一个完整的实时计算机视觉应用，展示了:
- 深度学习模型的实际部署
- 实时视频处理技术
- 用户友好的界面设计
- 工程级别的代码结构

整个系统体现了从模型训练到实际应用的完整流程，
是深度学习在计算机视觉领域的典型应用案例。
"""
