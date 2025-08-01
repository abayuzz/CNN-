# 🤖 深层CNN手势识别系统 v2.0

## 📋 项目简介
本项目使用**深层卷积神经网络**实现高精度的石头、剪刀、布手势识别。通过4层CNN架构、批归一化、智能训练策略等技术，显著提升了识别准确率和系统稳定性。

## 📁 文件结构
- `cnn.py`: **深层CNN模型**定义、训练、测试和保存代码（最新版本）
- `real_time_detection_deep.py`: **深层CNN实时检测**系统（推荐使用）
- `model_comparison.py`: 模型性能对比测试脚本
- `real_time_detection.py`: 原版实时检测脚本
- `create_enhanced_data.py`: 生成增强训练和测试数据
- `image_classifier.py`: 演示单张图片的分类
- `IMPROVEMENTS.md`: 详细的改进说明文档

## 🚀 快速开始（推荐流程）

### 1. 安装依赖
```bash
pip install torch torchvision numpy pillow opencv-python
```

### 2. 训练深层CNN模型
```bash
python cnn.py
```
该脚本会：
- 自动创建增强的示例数据（每类30张训练图，10张测试图）
- 训练4层深层CNN网络（参数量: ~50M）
- 实时显示训练进度和准确率
- 自动保存最佳模型到 `best_cnn_model.pth`

### 3. 运行深层CNN实时检测
```bash
python real_time_detection_deep.py
```
特色功能：
- 🎯 智能检测区域（右上角绿框）
- 📊 实时概率显示条
- ⏱️ FPS性能监控
- 📸 按's'保存截图
- 🔄 时间平滑算法减少抖动

### 4. 性能对比测试（可选）
```bash
python model_comparison.py
```

## 🏆 核心技术特性

### 🧠 深层CNN架构
```
输入 (3x224x224)
    ↓
Conv1 (32) + BatchNorm + MaxPool + Dropout(0.25)
    ↓
Conv2 (64) + BatchNorm + MaxPool + Dropout(0.25)
    ↓
Conv3 (128) + BatchNorm + MaxPool + Dropout(0.25)
    ↓
Conv4 (256) + BatchNorm + MaxPool + Dropout(0.3)
    ↓
FC1 (512) + BatchNorm + Dropout(0.5)
    ↓
FC2 (256) + BatchNorm + Dropout(0.5)
    ↓
FC3 (128) + BatchNorm + Dropout(0.3)
    ↓
输出 (3类)
```

### 📈 智能训练策略
- **自适应学习率**: 0.0001起始，每30轮衰减50%
- **早停机制**: 20轮无改进自动停止
- **梯度裁剪**: 防止梯度爆炸
- **权重衰减**: L2正则化系数1e-4

### 🎯 增强实时检测
- **时间平滑**: 7帧历史加权平均 `[0.1, 0.2, 0.3, 0.4]`
- **置信度阈值**: 75%减少误判
- **GPU加速**: 自动检测CUDA可用性
- **预处理优化**: 高斯模糊+ImageNet标准化

## 📊 性能对比

| 特性 | 原版CNN | 深层CNN | 改进幅度 |
|------|---------|---------|----------|
| 卷积层数 | 2层 | 4层 | +100% |
| 参数量 | ~25M | ~50M | +100% |
| 批归一化 | ❌ | ✅ | 全层支持 |
| Dropout | ❌ | ✅ | 多层正则化 |
| 学习率调度 | ❌ | ✅ | 自适应衰减 |
| 时间平滑 | ❌ | ✅ | 7帧加权 |
| 预期准确率 | ~70% | ~85%+ | +15%+ |

## 🛠️ 使用说明

### 训练自定义模型
如果要使用真实手势数据：
1. 按以下结构准备数据：
```
data/
  train/
    rock/     (石头图片)
    paper/    (布图片)
    scissors/ (剪刀图片)
  test/
    rock/     (测试图片)
    paper/    (测试图片)
    scissors/ (测试图片)
```
2. 确保每类至少30张训练图片
3. 运行 `python cnn.py`

### 实时检测技巧
- ✅ 确保充足的光照条件
- ✅ 将手势放在绿色检测框内
- ✅ 保持手势稳定1-2秒获得最佳效果
- ✅ 背景尽量简洁，避免复杂图案
- ⚠️ 避免快速移动手势

## 🔧 参数调优指南

### 提高识别准确率
```python
# 在 real_time_detection_deep.py 中调整
self.confidence_threshold = 0.8  # 提高置信度阈值
self.history_size = 10           # 增加时间平滑窗口
```

### 提高响应速度
```python
self.confidence_threshold = 0.6  # 降低置信度阈值
self.history_size = 5            # 减少平滑窗口
```

### 训练参数调整
```python
# 在 cnn.py 中调整
lr=0.0005                       # 提高学习率
weight_decay=5e-4               # 增强正则化
patience=30                     # 增加早停耐心
```

## 🐛 故障排除

### 常见问题及解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 模型加载失败 | 模型文件不存在 | 先运行 `python cnn.py` 训练模型 |
| 摄像头无法打开 | 权限或占用问题 | 检查摄像头连接，关闭其他摄像头应用 |
| 识别准确率低 | 光照/背景干扰 | 改善光照条件，使用简洁背景 |
| 程序运行慢 | CPU推理 | 安装CUDA版PyTorch启用GPU加速 |
| 内存不足 | 模型参数过多 | 减小batch_size或使用更小模型 |

### 性能调优命令
```bash
# 查看GPU使用情况
nvidia-smi

# 监控系统资源
htop

# 测试摄像头
python -c "import cv2; print('摄像头可用' if cv2.VideoCapture(0).isOpened() else '摄像头不可用')"
```

## 📈 进阶优化建议

### 1. 数据集改进
- 收集真实手势图片（建议每类500+张）
- 增加不同光照条件的样本
- 包含不同肤色、年龄的手势
- 添加复杂背景的训练数据

### 2. 模型架构优化
- 尝试ResNet、EfficientNet等现代架构
- 使用预训练模型进行迁移学习
- 实验不同的激活函数（Swish、GELU）
- 添加注意力机制

### 3. 训练策略改进
- 使用CosineAnnealingLR学习率调度
- 实验不同的优化器（AdamW、SGD）
- 添加数据增强技术（MixUp、CutMix）
- 使用交叉验证评估模型

## 📞 技术支持

如遇到问题，请检查：
1. Python版本 >= 3.8
2. PyTorch版本 >= 1.8
3. 摄像头驱动是否正常
4. 是否有足够的磁盘空间

## 📄 版本历史

- **v2.0** (2025-07-29): 深层CNN架构，智能训练，增强检测
- **v1.0** (2025-07-24): 基础CNN模型，简单实时检测

## 📜 许可证
本项目遵循 MIT 许可证。

---
*🎯 深层CNN让手势识别更准确、更稳定！*
