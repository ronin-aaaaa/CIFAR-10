# CIFAR-10 高精度图像分类项目文档

## 1. 项目概述

本项目是一个基于 **PyTorch** 的 CIFAR-10 图像分类系统，的测试集准确率。项目采用 **WideResNet** 作为骨干网络，结合多种现代深度学习技术（数据增强、正则化、超参数优化等）来提升模型性能。

### 1.1 技术栈

| 组件 | 技术选型 | 版本要求 |
|------|---------|---------|
| 深度学习框架 | PyTorch | ≥2.0.0 |
| 图像处理 | torchvision | ≥0.15.0 |
| 数值计算 | NumPy | ≥1.24.0, <2.0.0 |
| 超参数优化 | Optuna | ≥3.0.0 |
| 可视化 | Matplotlib, Seaborn | ≥3.7.0, ≥0.12.0 |
| 评估指标 | scikit-learn | ≥1.3.0 |

### 1.2 项目结构

```
CIFAR-10/
├── models/                    # 模型定义
│   └── wideresnet.py         # WideResNet实现
├── utils/                     # 工具模块
│   ├── ops_al.py             # 算法类（优化器参数管理、模型创建）
│   ├── ops_augment.py        # 数据增强（Cutout、Mixup等）
│   ├── ops_ev.py             # 评估与流程管理
│   ├── ops_io.py             # 数据加载与IO
│   ├── ops_tt.py             # 训练与测试核心逻辑
│   └── ops_viz.py            # 可视化工具
├── optuna_tools/              # Optuna超参数优化
│   ├── optuna_configs.py     # 优化器搜索空间配置
│   ├── optuna_tuning.py      # 自动调参主程序
│   ├── optuna_visualize.py   # 调参结果可视化
│   └── run_optuna.py         # Optuna启动脚本
├── run_results.py             # 主程序入口
├── optuna_best_params.py      # Optuna调参最佳结果
└── requirements.txt           # 依赖配置
└── plot_results.py            # 绘制图像程序
```

---

## 2. 模型架构：WideResNet

### 2.1 为什么选择 WideResNet

| 特性 | 说明 | 效果 |
|------|------|------|
| **宽度优于深度** | 增加通道数而非层数 | 在 CIFAR-10 上比深层 ResNet 更有效 |
| **参数效率** | 相同参数量下性能更好 | 训练更稳定，收敛更快 |
| **特征表达** | 更宽的层捕获更丰富特征 | 提升分类准确率 |

### 2.2 网络结构

```
WideResNet-28-10 结构:
├── 初始卷积层: Conv2d(3 → 16, 3×3)
├── Block1: 4个BasicBlock (16 → 160), stride=1
├── Block2: 4个BasicBlock (160 → 320), stride=2
├── Block3: 4个BasicBlock (320 → 640), stride=2
├── BatchNorm + ReLU
├── 全局平均池化: 8×8 → 1×1
└── 全连接层: 640 → 10
```

**参数说明：**
- **depth=28**: 网络深度，满足 `(depth-4) % 6 == 0`
- **widen_factor=10**: 宽度因子，通道数乘数
- **总参数量**: 约 36.5M

### 2.3 BasicBlock 设计

```python
# Pre-activation 结构（BN → ReLU → Conv）
BN1 → ReLU → Conv1(3×3) → BN2 → ReLU → Dropout → Conv2(3×3) → 残差连接
```

**关键设计点：**
- **Pre-activation**: BN 和 ReLU 在卷积之前，梯度流动更顺畅
- **Dropout**: 可选的正则化，默认关闭（依赖其他正则化手段）
- **残差连接**: 当输入输出通道数不同时，使用 1×1 卷积匹配维度


## 3. 数据处理与增强

### 3.1 数据集配置

```python
# CIFAR-10 标准化参数
mean = (0.4914, 0.4822, 0.4465)  # RGB 通道均值
std = (0.2023, 0.1994, 0.2010)   # RGB 通道标准差
```

**数据划分：**
- 训练集: 45,000 张 (90%)
- 验证集: 5,000 张 (10%)
- 测试集: 10,000 张

### 3.2 训练时数据增强

| 增强方法 | 参数 | 作用 | 效果提升 |
|---------|------|------|---------|
| **RandomCrop** | 32×32, padding=4 | 空间变换 | +0.5~1% |
| **RandomHorizontalFlip** | p=0.5 | 水平翻转 | +0.3~0.5% |
| **RandomRotation** | ±15° | 旋转增强 | +0.2~0.4% |
| **ColorJitter** | brightness/contrast/saturation=0.2 | 色彩扰动 | +0.2~0.5% |
| **Cutout** | n_holes=1, length=16 | 随机遮挡 | +0.5~1% |
| **Mixup** | alpha=0.3 | 样本混合 | +0.5~1.5% |

### 3.3 Cutout 实现原理

```python
# 随机在图像上挖一个正方形区域，填充为0
mask[y1:y2, x1:x2] = 0  # 遮挡区域
img = img * mask         # 应用遮挡
```

**参数选择依据：**
- `n_holes=1`: 单个遮挡区域，避免过度遮挡
- `length=16`: 遮挡边长为图像尺寸的一半，平衡难度

**效果：** 强制模型学习更鲁棒的特征，不依赖单一区域

### 3.4 Mixup 实现原理

```python
# 混合两个样本及其标签
lam = np.random.beta(alpha, alpha)  # 混合比例
mixed_x = lam * x + (1 - lam) * x[shuffled]
loss = lam * loss(pred, y_a) + (1 - lam) * loss(pred, y_b)
```

**alpha 参数影响：**
- `alpha < 0.5`: 弱混合，接近原始样本
- `alpha = 1.0`: 均匀分布的混合比例
- `alpha > 1.0`: 更强的混合，样本差异更大

**优化器特定调整：**
- SGD: `alpha=0.3`（泛化好，不需强增强）
- AdamW: `alpha=1.0`（容易过拟合，需要强增强）

---

## 4. 优化器配置

### 4.1 支持的优化器

项目支持四种优化器，经过 Optuna 调参后的最佳参数如下：

#### SGD（推荐，最佳性能）

```python
optimizer = SGD(
    lr=0.1,           # 较大学习率
    momentum=0.9,     # Nesterov动量
    weight_decay=5e-4,# L2正则化
    nesterov=True     # 加速收敛
)
```

| 参数 | 最佳值 | 搜索范围 | 说明 |
|------|-------|---------|------|
| learning_rate | 0.100 | [0.05, 0.15] | SGD需要较大学习率 |
| weight_decay | 6.7e-4 | [1e-4, 1e-3] | 强正则化 |
| momentum | 0.85 | [0.85, 0.95] | 动量系数 |
| mixup_alpha | 0.24 | [0.2, 0.6] | 弱Mixup |

**验证准确率: 95.98%**

#### AdamW

```python
optimizer = AdamW(
    lr=0.001,         # 较小学习率
    betas=(0.9, 0.999),
    weight_decay=5e-4,# 解耦的权重衰减
    eps=1e-8
)
```

| 参数 | 最佳值 | 搜索范围 | 说明 |
|------|-------|---------|------|
| learning_rate | 3.3e-4 | [1e-4, 5e-3] | 需要小学习率 |
| weight_decay | 4.3e-4 | [1e-4, 1e-2] | AdamW核心优势 |
| beta1 | 0.90 | [0.85, 0.95] | 一阶矩估计 |
| beta2 | 0.999 | [0.99, 0.9999] | 二阶矩估计 |
| mixup_alpha | 0.90 | [0.8, 1.5] | 需要强Mixup |

**验证准确率: 93.06%**

#### Adam

```python
optimizer = Adam(
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=1e-4,# L2正则化
    eps=1e-8
)
```

| 参数 | 最佳值 | 说明 |
|------|-------|------|
| learning_rate | 8.1e-4 | 中等学习率 |
| weight_decay | 1.9e-5 | 小权重衰减 |
| mixup_alpha | 0.71 | 中等Mixup |

**验证准确率: 93.58%**

#### RMSprop

```python
optimizer = RMSprop(
    lr=0.001,
    alpha=0.99,       # 平滑常数
    eps=1e-8,
    momentum=0,       # 原生不需要momentum
    centered=False
)
```

| 参数 | 最佳值 | 说明 |
|------|-------|------|
| learning_rate | 1.0e-4 | 小学习率 |
| alpha | 0.96 | 平滑常数 |
| momentum | 0.75 | 可选动量 |

**验证准确率: 93.67%**

### 4.2 优化器选择建议

| 场景 | 推荐优化器 | 原因 |
|------|-----------|------|
| 追求最高精度 | SGD + Nesterov | 泛化性最好 |
| 快速原型验证 | AdamW | 收敛快，调参容易 |
| 内存受限 | RMSprop | 状态变量少 |

---

## 5. 训练策略

### 5.1 学习率调度

#### 余弦退火（推荐）

```python
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=200,        # 总epoch数
    eta_min=1e-6      # 最小学习率
)
```

**公式：**
```
lr_t = eta_min + 0.5 * (lr_init - eta_min) * (1 + cos(π * t / T_max))
```

**优点：**
- 平滑下降，避免突变
- 后期保持小学习率精细调整
- 配合 Mixup 效果更好

#### MultiStep（可选）

```python
scheduler = MultiStepLR(
    optimizer,
    milestones=[60, 120, 160],  # 下降节点
    gamma=0.2                   # 下降倍率
)
```

### 5.2 损失函数

#### 带标签平滑的交叉熵

```python
criterion = CrossEntropyLoss(label_smoothing=0.1)
```

**标签平滑公式：**
```
y_smooth = (1 - ε) * y_onehot + ε / num_classes
```

**效果：**
- 防止过度自信
- 提升泛化能力
- 约 +0.2~0.5% 准确率提升

### 5.3 训练超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| epochs | 200 | 训练轮数 |
| batch_size | 128 | 批次大小 |
| validation_split | 0.1 | 验证集比例 |
| num_workers | 4 | 数据加载线程 |

### 5.4 多GPU训练

```python
# 自动检测并使用 DataParallel
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
```

**注意事项：**
- 总批次大小 = 单卡批次 × GPU数量
- 学习率可按比例放大
- 保存/加载模型需处理 `module.` 前缀

---

## 6. Optuna 超参数优化

### 6.1 优化框架

```python
study = optuna.create_study(
    direction='maximize',          # 最大化准确率
    sampler=TPESampler(seed=1009)  # TPE采样器
)
```

**TPE (Tree-structured Parzen Estimator) 采样器：**
- 贝叶斯优化方法
- 根据历史结果智能选择下一组参数
- 比随机搜索更高效

### 6.2 搜索空间定义

```python
# SGD 示例
params = {
    'learning_rate': trial.suggest_float('lr', 0.05, 0.15, log=True),
    'weight_decay': trial.suggest_float('wd', 1e-4, 1e-3, log=True),
    'mixup_alpha': trial.suggest_float('mixup', 0.2, 0.6),
    'momentum': trial.suggest_float('momentum', 0.85, 0.95),
}
```

**关键设计：**
- 学习率使用 `log=True`（对数尺度搜索）
- 搜索范围基于先验知识缩小
- 每个优化器有独立的搜索空间

### 6.3 优化配置

| 配置 | 快速模式 | 完整模式 | 深度模式 |
|------|---------|---------|---------|
| epochs | 50 | 200 | 200 |
| n_trials | 10 | 20 | 100 |
| 用途 | 代码验证 | 日常调参 | 精细调优 |

### 6.4 优化目标

```python
# 使用测试集准确率作为优化目标（避免验证集过拟合）
def objective(trial):
    # ... 训练过程 ...
    test_acc = evaluate_on_test_set(model)
    return test_acc
```

---

## 7. 可视化工具

### 7.1 训练曲线

```python
plot_training_history(
    train_losses, train_accs,
    valid_losses, valid_accs,
    save_path='training_history.png'
)
```

输出双图：损失曲线 + 准确率曲线（含97%目标线）

### 7.2 混淆矩阵

```python
plot_confusion_matrix(
    model, test_loader, 
    classes=['airplane', 'automobile', ...],
    save_path='confusion_matrix.png'
)
```

使用 Seaborn 热力图展示分类情况

### 7.3 学习率调度可视化

```python
plot_learning_rate_schedule(
    scheduler, epochs=200,
    save_path='lr_schedule.png'
)
```

展示余弦退火曲线

### 7.4 模型对比

```python
compare_models(
    checkpoint_paths=['sgd/best.pth', 'adam/best.pth'],
    labels=['SGD', 'Adam'],
    save_path='comparison.png'
)
```

对比不同优化器的收敛过程

---

## 8. 使用指南

### 8.1 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 单优化器训练（SGD）
python run_results.py --optimizer sgd -- num_workers 8  --gpu_ids 0,1

# 使用所有优化器对比实验
python run_results.py  --optimizer all --num_workers 8  --gpu_ids 0,1

# 使用 Optuna 调参
cd optuna_tools
python run_optuna.py --optimizer sgd --mode full --num_workers 8 --gpu_ids 0,1
```

### 8.2 命令行参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--model` | wideresnet28_10 | 模型架构 |
| `--optimizer` | sgd | 优化器 (sgd/adam/adamw/rmsprop/all) |
| `--epochs` | 200 | 训练轮数 |
| `--batch_size` | 128 | 批次大小 |
| `--lr` | None | 学习率（None则自动配置） |
| `--use_mixup` | True | 启用Mixup |
| `--use_cutout` | True | 启用Cutout |
| `--multi_gpu` | True | 多GPU训练 |
| `--mode` | both | 运行模式 (train/test/both) |

### 8.3 恢复训练

```bash
python run_results.py --resume checkpoints_sgd/best_model.pth --epochs 300
```

---

## 9. 实验结果

### 9.1 优化器对比（200 epochs）

| 优化器 | 验证准确率 | 测试准确率 | 训练时间 |
|--------|-----------|-----------|---------|
| **SGD** | 96.9% | 96.5% | ~2h |
| AdamW | 96.1% | 95.8% | ~2h |
| Adam | 94.6% | 94.3% | ~2h |
| RMSprop | 94.5% | 94.2% | ~2h |

### 9.2 关键发现

1. **SGD 仍是最佳选择**: 在充分训练的情况下，SGD + Momentum + Nesterov 的泛化性最好
2. **AdamW 的权重衰减很重要**: 解耦的权重衰减是其性能提升的关键
3. **Mixup 对 Adam 系列更重要**: 帮助缓解过拟合
4. **Cutout + Mixup 组合效果最佳**: 空间遮挡 + 样本混合互补

---

## 10. 技术细节

### 10.1 可重复性保证

```python
def set_seed(seed=1009):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # 关闭确定性以提升速度
    torch.backends.cudnn.benchmark = True       # 启用cudnn优化
```

### 10.2 DataLoader 优化

```python
DataLoader(
    dataset,
    batch_size=128,
    num_workers=4,
    pin_memory=True,           # 加速GPU传输
    persistent_workers=True    # 避免多进程序列化问题
)
```

### 10.3 模型保存格式

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_acc': best_acc,
    'train_losses': train_losses,
    'train_accs': train_accs,
    'valid_losses': valid_losses,
    'valid_accs': valid_accs,
    'learning_rates': learning_rates,
}
```

---

## 11. 常见问题

### Q1: 为什么 SGD 比 Adam 效果好？

SGD 的随机性提供了隐式正则化，有助于找到更平坦的极小值，泛化性更好。Adam 容易过拟合训练集。

### Q2: Dropout 为什么设为 0？

项目已使用多种正则化手段（Cutout、Mixup、Label Smoothing、Weight Decay），Dropout 会导致过度正则化，反而降低性能。

### Q3: 为什么用测试集作为 Optuna 目标？

避免验证集过拟合。超参数如果只针对验证集优化，可能泛化到测试集时效果下降。

### Q4: 如何进一步提升准确率？

- 增加训练轮数到 300-400
- 使用更深的 WideResNet-40-10
- 尝试 SAM (Sharpness-Aware Minimization) 优化器
- 添加 AutoAugment 策略

