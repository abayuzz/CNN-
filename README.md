# Rock-Paper-Scissors Hand Gesture Recognition

## 项目简介
本项目使用卷积神经网络 (CNN) 实现了一个基于手势识别的石头、剪刀、布游戏。通过实时视频流捕捉手势并进行分类，用户可以与系统进行交互。

## 文件结构
- `cnn.py`: 包含模型定义、训练、测试和保存的代码。
- `real_time_detection.py`: 实现实时手势检测和分类。
- `create_data.py`: 用于生成虚拟训练和测试数据。
- `image_classifier.py`: 演示单张图片的分类。

## 数据集结构
数据集应按照以下结构组织：
```
data/
  train/
    rock/     (石头图片)
    paper/    (布图片)
    scissors/ (剪刀图片)
  test/
    rock/     (石头图片)
    paper/    (布图片)
    scissors/ (剪刀图片)
```

## 环境依赖
- Python 3.8+
- PyTorch
- torchvision
- numpy
- PIL
- OpenCV

## 安装依赖
运行以下命令安装所需的 Python 库：
```bash
pip install torch torchvision numpy pillow opencv-python
```

## 使用说明
### 训练模型
运行以下命令训练模型：
```bash
python cnn.py
```
训练完成后，模型会保存到 `cnn_model.pth`。

### 实时手势检测
运行以下命令启动实时手势检测：
```bash
python real_time_detection.py
```

### 单张图片分类
运行以下命令对单张图片进行分类：
```bash
python image_classifier.py --image <图片路径>
```

## 注意事项
- 确保数据集已按照上述结构准备好。
- 如果没有实际数据集，可以使用 `create_data.py` 生成虚拟数据。

## 项目作者
- **作者**: [abayuzz]
- **日期**: 2025年7月24日

## 许可证
本项目遵循 MIT 许可证。
