"""
模型对比测试脚本
用于比较原版CNN和深层CNN的性能差异
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time
import os

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

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_performance(model, test_loader, device, model_name):
    """测试模型性能"""
    model.eval()
    correct = 0
    total = 0
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    inference_times = []
    
    print(f"\n测试 {model_name} 性能...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 测量推理时间
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 统计每个类别的准确率
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    overall_accuracy = 100 * correct / total
    avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒
    
    print(f"总体准确率: {overall_accuracy:.2f}%")
    print(f"平均推理时间: {avg_inference_time:.2f}ms")
    
    class_names = ['paper', 'rock', 'scissors']
    print("各类别准确率:")
    for i in range(3):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f"  {class_names[i]}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return overall_accuracy, avg_inference_time

def main():
    print("=" * 70)
    print("CNN模型性能对比测试")
    print("=" * 70)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据预处理
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载测试数据
    if not os.path.exists('data/test'):
        print("❌ 测试数据不存在，请先运行 cnn.py 生成数据")
        return
    
    test_dataset = datasets.ImageFolder(root='data/test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"测试数据: {len(test_dataset)} 张图片")
    
    # 初始化模型
    simple_model = SimpleCNN().to(device)
    deep_model = DeepCNN().to(device)
    
    # 打印模型信息
    print(f"\n模型参数对比:")
    print(f"简单CNN参数数量: {count_parameters(simple_model):,}")
    print(f"深层CNN参数数量: {count_parameters(deep_model):,}")
    print(f"参数增加倍数: {count_parameters(deep_model) / count_parameters(simple_model):.1f}x")
    
    # 测试简单CNN（如果模型文件存在）
    simple_accuracy = None
    simple_time = None
    if os.path.exists('simple_cnn_model.pth'):
        try:
            simple_model.load_state_dict(torch.load('simple_cnn_model.pth', map_location=device))
            simple_accuracy, simple_time = test_model_performance(
                simple_model, test_loader, device, "简单CNN"
            )
        except:
            print("无法加载简单CNN模型")
    else:
        print("\n简单CNN模型文件不存在，跳过测试")
    
    # 测试深层CNN
    deep_accuracy = None
    deep_time = None
    
    # 尝试加载最佳模型或普通模型
    model_loaded = False
    if os.path.exists('best_cnn_model.pth'):
        try:
            deep_model.load_state_dict(torch.load('best_cnn_model.pth', map_location=device))
            print("✓ 加载最佳深层CNN模型")
            model_loaded = True
        except:
            print("无法加载最佳深层CNN模型")
    
    if not model_loaded and os.path.exists('cnn_model.pth'):
        try:
            deep_model.load_state_dict(torch.load('cnn_model.pth', map_location=device))
            print("✓ 加载深层CNN模型")
            model_loaded = True
        except:
            print("无法加载深层CNN模型")
    
    if model_loaded:
        deep_accuracy, deep_time = test_model_performance(
            deep_model, test_loader, device, "深层CNN"
        )
    else:
        print("\n❌ 深层CNN模型文件不存在，请先运行 cnn.py 训练模型")
    
    # 性能对比总结
    print("\n" + "=" * 70)
    print("性能对比总结")
    print("=" * 70)
    
    print(f"{'模型':<15} {'参数数量':<15} {'准确率':<15} {'推理时间':<15} {'状态'}")
    print("-" * 70)
    
    simple_status = "✓ 可用" if simple_accuracy else "❌ 不可用"
    deep_status = "✓ 可用" if deep_accuracy else "❌ 不可用"
    
    print(f"{'简单CNN':<15} {count_parameters(simple_model):<15,} "
          f"{simple_accuracy or 'N/A':<15} {simple_time or 'N/A':<15} {simple_status}")
    
    print(f"{'深层CNN':<15} {count_parameters(deep_model):<15,} "
          f"{deep_accuracy or 'N/A':<15} {deep_time or 'N/A':<15} {deep_status}")
    
    # 改进效果分析
    if simple_accuracy and deep_accuracy:
        accuracy_improvement = deep_accuracy - simple_accuracy
        speed_ratio = simple_time / deep_time if simple_time and deep_time else None
        
        print(f"\n改进效果:")
        print(f"准确率提升: {accuracy_improvement:+.2f}%")
        if speed_ratio:
            if speed_ratio > 1:
                print(f"速度变化: 慢了 {1/speed_ratio:.1f}x")
            else:
                print(f"速度变化: 快了 {speed_ratio:.1f}x")
        
        print(f"\n建议:")
        if accuracy_improvement > 5:
            print("✓ 深层CNN在准确率上有显著提升，推荐使用")
        elif accuracy_improvement > 0:
            print("✓ 深层CNN准确率有所提升")
        else:
            print("⚠ 深层CNN准确率提升不明显，可能需要更多训练数据")
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()
