import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import time

# 定义与训练时相同的CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 3)  # 输出为3类，分别是石头、剪刀、布

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RealTimeGestureDetector:
    def __init__(self, model_path='cnn_model.pth'):
        # 类别标签
        self.class_names = ['paper', 'rock', 'scissors']  # 根据ImageFolder的字母顺序
        
        # 初始化模型
        self.model = CNN()
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            print("模型加载成功！")
        except FileNotFoundError:
            print(f"错误：找不到模型文件 {model_path}")
            print("请先运行 cnn.py 训练并保存模型")
            return
        
        # 数据预处理（与训练时保持一致）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("错误：无法打开摄像头")
            return
        
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("摄像头初始化成功！")
        print("按 'q' 退出程序")
    
    def preprocess_frame(self, frame):
        """预处理视频帧"""
        # 转换为PIL图像
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 应用变换
        tensor_image = self.transform(pil_image).unsqueeze(0)  # 添加批次维度
        
        return tensor_image
    
    def predict(self, frame):
        """预测手势"""
        # 预处理图像
        input_tensor = self.preprocess_frame(frame)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.class_names[predicted.item()]
            confidence_score = confidence.item()
            
        return predicted_class, confidence_score
    
    def draw_detection_area(self, frame):
        """绘制检测区域"""
        height, width = frame.shape[:2]
        
        # 定义检测区域（右上角区域）
        x1, y1 = width - 350, 30  # 扩大宽度和高度
        x2, y2 = width - 30, 330
        
        # 绘制检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Detection Area", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame[y1:y2, x1:x2]  # 返回检测区域
    
    def run(self):
        """运行实时检测"""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return
        
        print("开始实时检测...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 翻转图像（镜像效果）
            #frame = cv2.flip(frame, 1)
            
            # 绘制检测区域并获取ROI
            detection_roi = self.draw_detection_area(frame)
            
            try:
                # 预测手势
                gesture, confidence = self.predict(detection_roi)
                
                # 设置置信度阈值
                if confidence > 0.5:
                    # 显示预测结果
                    result_text = f"{gesture}: {confidence:.2f}"
                    color = (0, 255, 0)  # 绿色
                else:
                    result_text = "未识别"
                    color = (0, 0, 255)  # 红色
                
                # 在图像上显示结果
                cv2.putText(frame, result_text, (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
                # 显示所有类别的概率
                y_offset = 100
                with torch.no_grad():
                    input_tensor = self.preprocess_frame(detection_roi)
                    outputs = self.model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    
                    for i, class_name in enumerate(self.class_names):
                        prob_text = f"{class_name}: {probabilities[i]:.3f}"
                        cv2.putText(frame, prob_text, (50, y_offset + i*30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            except Exception as e:
                cv2.putText(frame, "Detection Error", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 显示帮助信息
            cv2.putText(frame, "Press 'q' to quit", (50, frame.shape[0]-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示画面
            cv2.imshow('Rock Paper Scissors Detection', frame)
            
            # 检查退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 清理资源
        self.cap.release()
        cv2.destroyAllWindows()
        print("程序结束")

def main():
    """主函数"""
    detector = RealTimeGestureDetector()
    detector.run()

if __name__ == "__main__":
    main()
