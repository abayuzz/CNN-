"""
石头剪刀布图片分类演示程序

此程序展示如何使用训练好的CNN模型对单张图片进行手势识别。
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# 定义CNN模型（与训练时相同）
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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

def load_model(model_path='cnn_model.pth'):
    """加载训练好的模型"""
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_gesture(model, image_path):
    """预测单张图片的手势"""
    # 类别标签
    class_names = ['paper', 'rock', 'scissors']
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # 加载并预处理图片
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # 模型预测
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
    
    return predicted_class, confidence_score, probabilities[0]

def create_test_image():
    """创建一个测试图片"""
    # 创建一个简单的测试图片
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # 添加一些简单的形状来模拟手势
    center_x, center_y = 112, 112
    
    # 在图片中央画一个圆形（模拟拳头）
    y, x = np.ogrid[:224, :224]
    mask = (x - center_x)**2 + (y - center_y)**2 < 50**2
    img_array[mask] = [255, 255, 255]  # 白色
    
    img = Image.fromarray(img_array)
    test_path = 'test_gesture.jpg'
    img.save(test_path)
    return test_path

def main():
    print("=== 石头剪刀布图片分类演示 ===")
    print()
    
    # 检查模型文件
    model_path = 'cnn_model.pth'
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        print("请先运行 'python cnn.py' 训练模型")
        return
    
    # 加载模型
    print("正在加载模型...")
    try:
        model = load_model(model_path)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 创建测试图片
    print("正在创建测试图片...")
    test_image_path = create_test_image()
    print(f"✓ 测试图片已创建: {test_image_path}")
    
    # 进行预测
    print("\n正在进行预测...")
    try:
        predicted_class, confidence, probabilities = predict_gesture(model, test_image_path)
        
        print(f"\n=== 预测结果 ===")
        print(f"预测类别: {predicted_class}")
        print(f"置信度: {confidence:.4f}")
        
        print(f"\n=== 各类别概率 ===")
        class_names = ['paper', 'rock', 'scissors']
        for i, class_name in enumerate(class_names):
            prob = probabilities[i].item()
            bar = "█" * int(prob * 20)  # 创建简单的进度条
            print(f"{class_name:8}: {prob:.4f} {bar}")
        
        print(f"\n✓ 预测完成")
        
    except Exception as e:
        print(f"✗ 预测失败: {e}")
    
    # 测试数据集中的图片
    print("\n=== 测试数据集样本 ===")
    test_folders = ['data/test/rock', 'data/test/paper', 'data/test/scissors']
    
    for folder in test_folders:
        if os.path.exists(folder):
            images = [f for f in os.listdir(folder) if f.endswith('.jpg')]
            if images:
                # 测试第一张图片
                image_path = os.path.join(folder, images[0])
                try:
                    predicted_class, confidence, _ = predict_gesture(model, image_path)
                    actual_class = os.path.basename(folder)
                    correct = "✓" if predicted_class == actual_class else "✗"
                    print(f"{correct} {image_path}: 预测={predicted_class}, 实际={actual_class}, 置信度={confidence:.3f}")
                except Exception as e:
                    print(f"✗ {image_path}: 预测失败 - {e}")
    
    print("\n=== 使用说明 ===")
    print("1. 将您的手势图片命名为任意名称，放在当前目录")
    print("2. 修改 main() 函数中的 test_image_path 变量")
    print("3. 重新运行程序即可看到预测结果")
    print("\n注意：由于使用虚拟数据训练，预测结果仅供演示")

if __name__ == "__main__":
    main()
