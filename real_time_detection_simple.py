"""
简化版实时手势识别系统
兼容原版和深层CNN模型
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import os

# 自动检测模型结构的CNN类
class AdaptiveCNN(nn.Module):
    def __init__(self):
        super(AdaptiveCNN, self).__init__()
        # 这里会在加载模型时自动适配结构
        pass

# 原版简单CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 深层CNN
class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        
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
        
        self.fc4 = nn.Linear(128, 3)

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

class SimpleGestureDetector:
    def __init__(self):
        print("初始化兼容版手势识别系统...")
        
        # 类别标签
        self.class_names = ['paper', 'rock', 'scissors']
        
        # 预测历史记录
        self.prediction_history = []
        self.history_size = 5  # 减少历史记录长度
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ 使用设备: {self.device}")
        
        # 尝试加载模型
        self.model = None
        self.model_type = None
        
        # 按优先级尝试加载不同的模型
        model_files = [
            ('best_cnn_model.pth', '最佳深层CNN'),
            ('cnn_model.pth', 'CNN模型'),
        ]
        
        for model_file, model_desc in model_files:
            if os.path.exists(model_file):
                if self.try_load_model(model_file, model_desc):
                    break
        
        if self.model is None:
            print("❌ 没有找到可用的模型文件!")
            print("请先运行 cnn.py 训练模型")
            return
        
        # 数据预处理
        if self.model_type == 'deep':
            # 深层CNN使用标准化
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # 简单CNN不使用标准化
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ 错误：无法打开摄像头")
            return
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✓ 摄像头初始化成功！")
        print("\n使用说明：")
        print("- 将手势放在绿色检测框内")
        print("- 按 'q' 退出程序")
    
    def try_load_model(self, model_file, model_desc):
        """尝试加载模型"""
        try:
            # 先尝试加载为深层CNN
            model = DeepCNN().to(self.device)
            model.load_state_dict(torch.load(model_file, map_location=self.device))
            model.eval()
            self.model = model
            self.model_type = 'deep'
            print(f"✓ {model_desc}加载成功 (深层CNN)")
            return True
        except:
            try:
                # 如果失败，尝试加载为简单CNN
                model = SimpleCNN().to(self.device)
                model.load_state_dict(torch.load(model_file, map_location=self.device))
                model.eval()
                self.model = model
                self.model_type = 'simple'
                print(f"✓ {model_desc}加载成功 (简单CNN)")
                return True
            except Exception as e:
                print(f"❌ 加载{model_desc}失败: {e}")
                return False
    
    def preprocess_frame(self, frame):
        """预处理视频帧"""
        # 转换为PIL图像
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 应用变换
        tensor_image = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        return tensor_image
    
    def predict_simple(self, frame):
        """简单预测，无复杂平滑"""
        # 预处理图像
        input_tensor = self.preprocess_frame(frame)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted = np.argmax(probabilities)
            
            predicted_class = self.class_names[predicted]
            confidence = probabilities[predicted]
            
            # 简单的历史记录平滑
            self.prediction_history.append(predicted_class)
            if len(self.prediction_history) > self.history_size:
                self.prediction_history.pop(0)
            
            # 使用最常见的预测作为最终结果
            if len(self.prediction_history) >= 3:
                from collections import Counter
                most_common = Counter(self.prediction_history).most_common(1)[0]
                if most_common[1] >= 2:  # 如果最常见的预测出现至少2次
                    predicted_class = most_common[0]
            
            return predicted_class, confidence, probabilities
    
    def draw_detection_area(self, frame):
        """绘制检测区域"""
        height, width = frame.shape[:2]
        
        # 定义检测区域
        x1, y1 = width - 300, 50
        x2, y2 = width - 50, 300
        
        # 绘制检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Hand Area", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame[y1:y2, x1:x2]
    
    def draw_ui(self, frame, gesture, confidence, probabilities):
        """绘制用户界面"""
        # 显示预测结果
        if confidence > 0.6:  # 降低阈值
            result_text = f"{gesture}: {confidence:.2f}"
            color = (0, 255, 0)
        else:
            result_text = "未识别"
            color = (0, 0, 255)
        
        cv2.putText(frame, result_text, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # 显示所有类别的概率
        y_offset = 100
        for i, class_name in enumerate(self.class_names):
            prob = probabilities[i]
            prob_text = f"{class_name}: {prob:.3f}"
            cv2.putText(frame, prob_text, (20, y_offset + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示模型类型
        cv2.putText(frame, f"Model: {self.model_type.upper()}", (20, frame.shape[0]-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self):
        """运行实时检测"""
        if self.model is None or not self.cap.isOpened():
            print("❌ 初始化未完成")
            return
        
        print("\n🚀 开始实时手势识别...")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 无法读取摄像头画面")
                    break
                
                # 翻转图像
                frame = cv2.flip(frame, 1)
                
                # 绘制检测区域并获取ROI
                detection_roi = self.draw_detection_area(frame)
                
                try:
                    # 预测手势
                    gesture, confidence, probabilities = self.predict_simple(detection_roi)
                    
                    # 绘制用户界面
                    self.draw_ui(frame, gesture, confidence, probabilities)
                    
                except Exception as e:
                    error_msg = f"预测错误: {str(e)[:25]}"
                    print(f"❌ {error_msg}")
                    cv2.putText(frame, error_msg, (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示画面
                cv2.imshow('兼容版手势识别系统', frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 用户退出程序")
                    break
        
        except KeyboardInterrupt:
            print("\n👋 程序被用户中断")
        
        finally:
            # 清理资源
            self.cap.release()
            cv2.destroyAllWindows()
            print("✓ 资源清理完成")

def main():
    """主函数"""
    print("=" * 50)
    print("兼容版手势识别系统")
    print("支持原版和深层CNN模型")
    print("=" * 50)
    
    detector = SimpleGestureDetector()
    detector.run()

if __name__ == "__main__":
    main()
