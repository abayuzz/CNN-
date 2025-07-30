import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# 定义改进的深层卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 第一个卷积块 - 提取基础特征
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # 第二个卷积块 - 提取中层特征
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # 第三个卷积块 - 提取高层特征
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # 第四个卷积块 - 提取更复杂特征
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout2d(0.3)
        
        # 全连接层 - 分类器
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

# 改进的数据预处理
# 训练时使用数据增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),              # 随机旋转 ±15度
    transforms.RandomHorizontalFlip(0.5),       # 50%概率水平翻转
    transforms.ColorJitter(                     # 颜色抖动
        brightness=0.2, 
        contrast=0.2, 
        saturation=0.2, 
        hue=0.1
    ),
    transforms.RandomAffine(                    # 随机仿射变换
        degrees=0, 
        translate=(0.1, 0.1), 
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(                       # ImageNet标准化
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 测试时不使用数据增强
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 加载数据集 (注意：需要准备实际的数据集)
# 数据集目录结构应该如下：
# data/
#   train/
#     rock/     (石头图片)
#     paper/    (布图片)  
#     scissors/ (剪刀图片)
#   test/
#     rock/     (石头图片)
#     paper/    (布图片)
#     scissors/ (剪刀图片)
"""
def create_dummy_data():
    # 创建数据目录
    for split in ['train', 'test']:
        for class_name in ['rock', 'paper', 'scissors']:
            os.makedirs(f'data/{split}/{class_name}', exist_ok=True)
            
            # 为每个类别创建一些虚拟图片
            for i in range(10):  # 每个类别10张图片
                # 创建随机彩色图片
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(f'data/{split}/{class_name}/{class_name}_{i}.jpg')
# 创建虚拟数据（仅用于测试）
# 注意：为了获得更好的效果，建议使用真实的手势图片数据
# create_dummy_data()
"""

# 使用不同的变换加载训练和测试数据
train_dataset = datasets.ImageFolder(root='data/train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 减小批次大小

test_dataset = datasets.ImageFolder(root='data/test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 初始化网络、损失函数和优化器
model = CNN()
criterion = nn.CrossEntropyLoss()
# 使用更小的学习率和权重衰减
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
# 添加学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 改进的训练函数
def train_model():
    model.train()
    best_accuracy = 0.0
    patience = 20  # 早停耐心值
    patience_counter = 0
    
    print("开始训练深层CNN模型...")
    
    for epoch in range(150):  # 增加训练轮数
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            
            # 计算训练准确率
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # 每5个epoch评估一次
        if (epoch + 1) % 5 == 0:
            train_accuracy = 100 * correct_train / total_train
            avg_loss = running_loss / len(train_loader)
            
            # 测试模型
            model.eval()
            correct_test = 0
            total_test = 0
            test_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
            
            test_accuracy = 100 * correct_test / total_test
            avg_test_loss = test_loss / len(test_loader)
            
            print(f'Epoch [{epoch+1}/150]')
            print(f'  训练: Loss={avg_loss:.4f}, Accuracy={train_accuracy:.2f}%')
            print(f'  测试: Loss={avg_test_loss:.4f}, Accuracy={test_accuracy:.2f}%')
            print(f'  学习率: {scheduler.get_last_lr()[0]:.6f}')
            print('-' * 50)
            
            # 保存最佳模型
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(model.state_dict(), 'best_cnn_model.pth')
                print(f'新的最佳模型已保存! 测试准确率: {best_accuracy:.2f}%')
                patience_counter = 0
            else:
                patience_counter += 1
            
            model.train()
        
        # 更新学习率
        scheduler.step()
        
        # 早停检查
        if patience_counter >= patience:
            print(f'早停触发! 最佳测试准确率: {best_accuracy:.2f}%')
            break
    
    print(f'训练完成! 最佳测试准确率: {best_accuracy:.2f}%')

# 改进的测试函数
def test_model():
    print("评估最终模型性能...")
    model.eval()
    correct = 0
    total = 0
    class_correct = [0, 0, 0]  # 每个类别的正确预测数
    class_total = [0, 0, 0]    # 每个类别的总样本数
    class_names = ['paper', 'rock', 'scissors']  # ImageFolder的默认排序
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 统计每个类别的准确率
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    overall_accuracy = 100 * correct / total
    print(f'\n总体测试准确率: {overall_accuracy:.2f}%')
    
    print('\n各类别准确率:')
    for i in range(3):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'  {class_names[i]}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})')
        else:
            print(f'  {class_names[i]}: 无测试样本')

def save_model():
    # 保存最终模型
    torch.save(model.state_dict(), 'cnn_model.pth')
    print("\n最终模型已保存到 cnn_model.pth")
    
    # 如果存在最佳模型，复制为最终模型
    if os.path.exists('best_cnn_model.pth'):
        import shutil
        shutil.copy('best_cnn_model.pth', 'cnn_model.pth')
        print("最佳模型已复制为最终模型")

def create_sample_data():
    """创建一些示例数据用于测试"""
    from PIL import Image
    
    print("创建示例数据...")
    for split in ['train', 'test']:
        for class_name in ['rock', 'paper', 'scissors']:
            os.makedirs(f'data/{split}/{class_name}', exist_ok=True)
            
            # 为训练集创建更多样本
            num_samples = 30 if split == 'train' else 10
            
            for i in range(num_samples):
                # 创建具有不同特征的图片
                img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
                
                # 为不同类别添加不同的模式
                if class_name == 'rock':
                    # 石头：添加圆形区域
                    center = (112, 112)
                    for x in range(224):
                        for y in range(224):
                            if (x-center[0])**2 + (y-center[1])**2 < 3600:
                                img_array[x, y] = [100, 50, 30]
                elif class_name == 'paper':
                    # 布：添加矩形区域
                    img_array[50:174, 60:164] = [150, 120, 100]
                elif class_name == 'scissors':
                    # 剪刀：添加两个条形区域
                    img_array[80:150, 70:90] = [140, 110, 90]
                    img_array[80:150, 134:154] = [140, 110, 90]
                
                img = Image.fromarray(img_array)
                img.save(f'data/{split}/{class_name}/{class_name}_{i:03d}.jpg')
    
    print("示例数据创建完成!")

def main():
    print("=" * 60)
    print("深层CNN手势识别训练程序")
    print("=" * 60)
    
    # 检查数据是否存在
    if not os.path.exists('data/train') or not os.path.exists('data/test'):
        print("数据目录不存在，正在创建示例数据...")
        create_sample_data()
    
    # 检查数据量
    train_count = sum([len(os.listdir(f'data/train/{cls}')) 
                      for cls in ['rock', 'paper', 'scissors'] 
                      if os.path.exists(f'data/train/{cls}')])
    test_count = sum([len(os.listdir(f'data/test/{cls}')) 
                     for cls in ['rock', 'paper', 'scissors'] 
                     if os.path.exists(f'data/test/{cls}')])
    
    print(f"训练数据: {train_count} 张图片")
    print(f"测试数据: {test_count} 张图片")
    
    if train_count == 0:
        print("警告: 没有找到训练数据!")
        return
    
    # 开始训练
    train_model()
    
    # 测试模型
    test_model()
    
    # 保存模型
    save_model()
    
    print("\n" + "=" * 60)
    print("训练完成! 现在可以运行 real_time_detection.py 进行实时检测")
    print("=" * 60)

if __name__ == "__main__":
    main()  # 执行训练和测试