"""
深层CNN实时手势识别系统
支持石头、剪刀、布手势的实时检测
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import os
from collections import Counter

# 定义与训练时相同的深层CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # 第四个卷积块
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout2d(0.3)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout6 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.dropout7 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(128, 3)  # 输出为3类：石头、剪刀、布

    def forward(self, x):
        # 第一个卷积块
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # 第二个卷积块
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        # 第三个卷积块
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        # 第四个卷积块
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)
        
        # 展平并通过全连接层
        x = x.view(-1, 256 * 14 * 14)
        
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout5(x)
        
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout6(x)
        
        x = torch.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout7(x)
        
        x = self.fc4(x)
        return x

class DeepGestureDetector:
    def __init__(self, model_path='cnn_model.pth'):
        print("初始化深层CNN手势识别系统...")
        
        # 类别标签
        self.class_names = ['paper', 'rock', 'scissors']  # ImageFolder默认排序
        self.class_names_cn = ['paper', 'rock', 'scissors']  # 中文显示
        
        # 预测历史记录，用于时间平滑
        self.prediction_history = []
        self.history_size = 7  # 增加历史记录长度
        self.confidence_threshold = 0.75  # 提高置信度阈值
        
        # 初始化模型
        self.model = CNN()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        try:
            # 优先加载最佳模型
            if os.path.exists('best_cnn_model.pth'):
                self.model.load_state_dict(torch.load('best_cnn_model.pth', map_location=self.device))
                print("✓ 最佳模型加载成功！")
            else:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("✓ 模型加载成功！")
            
            self.model.eval()
            print(f"✓ 使用设备: {self.device}")
            
        except FileNotFoundError:
            print(f"❌ 错误：找不到模型文件 {model_path}")
            print("请先运行 cnn.py 训练并保存模型")
            return
        except Exception as e:
            print(f"❌ 模型加载错误: {e}")
            return
        
        # 改进的数据预处理（与训练时保持一致）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ 错误：无法打开摄像头")
            return
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ 摄像头初始化成功！")
        print("\n使用说明：")
        print("- 将手势放在绿色检测框内")
        print("- 保持手势稳定1-2秒获得最佳识别效果")
        print("- 按 'q' 退出程序")
        print("- 按 's' 保存当前帧")
    
    def preprocess_frame(self, frame):
        """改进的预处理视频帧"""
        # 应用高斯模糊减少噪音
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 应用变换
        tensor_image = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        return tensor_image
    
    def predict_with_smoothing(self, frame):
        """带时间平滑的预测"""
        # 预处理图像
        input_tensor = self.preprocess_frame(frame)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu()
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.class_names[predicted.item()]
            confidence_score = confidence.item()
            
            # 添加到历史记录
            self.prediction_history.append((predicted_class, confidence_score, probabilities[0].numpy()))
            
            # 保留最近的预测结果
            if len(self.prediction_history) > self.history_size:
                self.prediction_history.pop(0)
            
            # 使用加权平均进行时间平滑
            if len(self.prediction_history) >= 3:
                # 根据历史记录长度动态生成权重
                history_len = len(self.prediction_history)
                
                if history_len == 3:
                    weights = np.array([0.2, 0.3, 0.5])
                elif history_len == 4:
                    weights = np.array([0.1, 0.2, 0.3, 0.4])
                elif history_len == 5:
                    weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
                elif history_len == 6:
                    weights = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.25])
                else:  # history_len == 7
                    weights = np.array([0.05, 0.1, 0.1, 0.15, 0.2, 0.2, 0.2])
                
                # 确保权重数组长度与历史记录匹配
                weights = weights[:history_len]
                weights = weights / weights.sum()  # 归一化权重
                
                avg_probs = np.average([item[2] for item in self.prediction_history], 
                                     weights=weights, axis=0)
                
                # 找到平均概率最高的类别
                smooth_predicted = np.argmax(avg_probs)
                smooth_confidence = avg_probs[smooth_predicted]
                smooth_class = self.class_names[smooth_predicted]
                
                return smooth_class, smooth_confidence, avg_probs
            
        return predicted_class, confidence_score, probabilities[0].numpy()
    
    def draw_detection_area(self, frame):
        """绘制检测区域"""
        height, width = frame.shape[:2]
        
        # 定义检测区域（右上角，更大的区域）
        x1, y1 = width - 380, 50
        x2, y2 = width - 50, 380
        
        # 绘制检测框（渐变边框效果）
        cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 150, 0), 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加标题
        cv2.putText(frame, "Hand Gesture Area", (x1, y1-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame[y1:y2, x1:x2]  # 返回检测区域
    
    def draw_ui(self, frame, gesture, confidence, probabilities):
        """绘制用户界面"""
        # 绘制背景面板
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 显示预测结果
        if confidence > self.confidence_threshold:
            gesture_cn = self.class_names_cn[self.class_names.index(gesture)]
            result_text = f"{gesture_cn} ({gesture})"
            color = (0, 255, 0)  # 绿色
            status = "识别成功"
        else:
            result_text = "未识别"
            color = (0, 165, 255)  # 橙色
            status = "置信度不足"
        
        # 主要结果显示
        cv2.putText(frame, result_text, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"置信度: {confidence:.1%}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"状态: {status}", (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示所有类别的概率
        y_offset = 135
        for i, (class_name, class_cn) in enumerate(zip(self.class_names, self.class_names_cn)):
            prob = probabilities[i]
            bar_width = int(prob * 200)
            
            # 绘制概率条
            cv2.rectangle(frame, (20, y_offset + i*20), (20 + bar_width, y_offset + i*20 + 15), 
                         (0, 255, 0) if i == np.argmax(probabilities) else (100, 100, 100), -1)
            cv2.rectangle(frame, (20, y_offset + i*20), (220, y_offset + i*20 + 15), 
                         (255, 255, 255), 1)
            
            # 概率文本
            prob_text = f"{class_cn}: {prob:.1%}"
            cv2.putText(frame, prob_text, (230, y_offset + i*20 + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示帮助信息
        help_y = frame.shape[0] - 60
        cv2.putText(frame, "按键: 'q'-退出, 's'-保存截图", (20, help_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f"FPS: {getattr(self, 'fps', 0):.1f}", (20, help_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self):
        """运行实时检测"""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            print("❌ 摄像头未正确初始化")
            return
        
        print("\n🚀 开始实时手势识别...")
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 无法读取摄像头画面")
                    break
                
                # 翻转图像（镜像效果，让用户感觉更自然）
                frame = cv2.flip(frame, 1)
                
                # 绘制检测区域并获取ROI
                detection_roi = self.draw_detection_area(frame)
                
                try:
                    # 预测手势
                    gesture, confidence, probabilities = self.predict_with_smoothing(detection_roi)
                    
                    # 绘制用户界面
                    self.draw_ui(frame, gesture, confidence, probabilities)
                    
                except Exception as e:
                    error_msg = f"检测错误: {str(e)[:30]}"
                    print(f"❌ {error_msg}")  # 在控制台也打印错误
                    cv2.putText(frame, error_msg, (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 显示更详细的错误信息
                    cv2.putText(frame, "请检查模型文件是否正确", (20, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # 计算FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    self.fps = 30 / elapsed if elapsed > 0 else 0
                    start_time = time.time()
                
                # 显示画面
                cv2.imshow('深层CNN手势识别系统', frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 用户退出程序")
                    break
                elif key == ord('s'):
                    # 保存截图
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"gesture_capture_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 截图已保存: {filename}")
        
        except KeyboardInterrupt:
            print("\n👋 程序被用户中断")
        
        finally:
            # 清理资源
            self.cap.release()
            cv2.destroyAllWindows()
            print("✓ 资源清理完成")

def main():
    """主函数"""
    print("=" * 60)
    print("深层CNN实时手势识别系统 v2.0")
    print("支持：石头、剪刀、布手势识别")
    print("=" * 60)
    
    detector = DeepGestureDetector()
    if hasattr(detector, 'cap') and detector.cap.isOpened():
        detector.run()
    else:
        print("❌ 初始化失败，程序退出")

if __name__ == "__main__":
    main()
