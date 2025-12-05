"""
可视化工具
用于绘制训练曲线和结果分析
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def plot_training_history(train_losses, train_accs, valid_losses, valid_accs, save_path='training_history.png'):
    """
    绘制训练历史曲线
    
    参数:
        train_losses: 训练损失列表
        train_accs: 训练准确率列表
        valid_losses: 验证损失列表
        valid_accs: 验证准确率列表
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 绘制损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, valid_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.axhline(y=97.0, color='g', linestyle='--', label='Target (97%)', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(model, test_loader, classes, device='cuda', save_path='confusion_matrix.png'):
    """
    绘制混淆矩阵
    
    参数:
        model: 模型
        test_loader: 测试数据加载器
        classes: 类别名称列表
        device: 设备
        save_path: 保存路径
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_learning_rate_schedule(scheduler, epochs, save_path='lr_schedule.png'):
    """
    绘制学习率调度曲线
    
    参数:
        scheduler: 学习率调度器
        epochs: 总轮数
        save_path: 保存路径
    """
    lrs = []
    for epoch in range(epochs):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), lrs, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Learning rate schedule plot saved to {save_path}")
    plt.close()


def visualize_augmentations(dataset, num_samples=8, save_path='augmentations.png'):
    """
    可视化数据增强效果
    
    参数:
        dataset: 数据集
        num_samples: 样本数量
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img, label = dataset[i]
        
        # 反标准化
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        # 转换为numpy并显示
        img_np = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Augmentation visualization saved to {save_path}")
    plt.close()


def load_and_plot_checkpoint(checkpoint_path, save_dir='./plots'):
    """
    从检查点加载并绘制训练历史
    
    参数:
        checkpoint_path: 检查点路径
        save_dir: 图表保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'train_losses' in checkpoint:
        plot_training_history(
            checkpoint['train_losses'],
            checkpoint['train_accs'],
            checkpoint['valid_losses'],
            checkpoint['valid_accs'],
            save_path=os.path.join(save_dir, 'training_history.png')
        )
        print(f"\nTraining Statistics:")
        print(f"  Epochs completed: {checkpoint['epoch']}")
        print(f"  Best validation accuracy: {checkpoint['best_acc']:.2f}%")
        print(f"  Final training accuracy: {checkpoint['train_accs'][-1]:.2f}%")
        print(f"  Final validation accuracy: {checkpoint['valid_accs'][-1]:.2f}%")
    else:
        print("No training history found in checkpoint")


def compare_models(checkpoint_paths, labels, save_path='model_comparison.png'):
    """
    比较多个模型的训练曲线
    
    参数:
        checkpoint_paths: 检查点路径列表
        labels: 模型标签列表
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(checkpoint_paths)))
    
    for i, (path, label) in enumerate(zip(checkpoint_paths, labels)):
        if not os.path.isfile(path):
            print(f"Checkpoint not found: {path}")
            continue
        
        checkpoint = torch.load(path, map_location='cpu')
        
        if 'train_losses' in checkpoint:
            epochs = range(1, len(checkpoint['train_losses']) + 1)
            
            ax1.plot(epochs, checkpoint['valid_losses'], 
                    color=colors[i], label=label, linewidth=2)
            ax2.plot(epochs, checkpoint['valid_accs'], 
                    color=colors[i], label=label, linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Model Comparison - Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax2.set_title('Model Comparison - Accuracy', fontsize=14, fontweight='bold')
    ax2.axhline(y=97.0, color='g', linestyle='--', label='Target (97%)', linewidth=2)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to {save_path}")
    plt.close()
