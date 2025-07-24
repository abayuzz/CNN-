import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# 定义卷积神经网络
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

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
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
create_dummy_data()
"""
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.ImageFolder(root='data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化网络、损失函数和优化器
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train_model():
    model.train()
    for epoch in range(250): 
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# 测试模型
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')

def save_model():
    torch.save(model.state_dict(), 'cnn_model.pth')
    print("Model saved to cnn_model.pth")

def main():
    # 执行训练和测试
    train_model()
    test_model()
    save_model()

if __name__ == "__main__":
    main()  # 执行训练和测试