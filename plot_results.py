"""
一键绘制训练结果对比图
自动检测所有优化器的checkpoint并生成对比图表
"""
import os
import glob
import torch
import argparse
import numpy as np
from utils.ops_viz import (
    compare_models, load_and_plot_checkpoint,
    plot_confusion_matrix, plot_learning_rate_schedule,
    visualize_augmentations
)
from utils.ops_io import CIFAR10DataLoader
from utils.ops_al import create_model
from torch.optim.lr_scheduler import CosineAnnealingLR

def plot_all_results(include_extra=True, include_confusion=True, 
                     include_lr_schedule=True, include_augmentation=True,
                     include_optuna=True):
    """自动绘制所有结果
    
    参数:
        include_extra: 是否生成额外的可视化（混淆矩阵、学习率曲线等）
        include_confusion: 是否生成混淆矩阵
        include_lr_schedule: 是否生成学习率调度曲线
        include_augmentation: 是否生成数据增强可视化
        include_optuna: 是否生成Optuna调参结果可视化
    """
    print("="*60)
    print("自动生成训练结果对比图")
    print("="*60)
    
    # 查找所有优化器的checkpoint
    checkpoint_dirs = glob.glob('checkpoints_*')
    
    if not checkpoint_dirs:
        print("\n未找到优化器训练结果目录 (checkpoints_*)")
        print("请先运行: python run_results.py --optimizer all")
        return
    
    checkpoint_paths = []
    labels = []
    
    for dir_name in sorted(checkpoint_dirs):
        best_model_path = os.path.join(dir_name, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint_paths.append(best_model_path)
            # 从目录名提取优化器名称
            optimizer_name = dir_name.replace('checkpoints_', '').upper()
            labels.append(optimizer_name)
            print(f"✓ 找到: {optimizer_name} - {best_model_path}")
    
    if not checkpoint_paths:
        print("\n未找到任何 best_model.pth 文件")
        return
    
    print(f"\n共找到 {len(checkpoint_paths)} 个优化器的训练结果")
    print("\n开始生成对比图表...")
    
    # 生成对比图
    compare_models(
        checkpoint_paths,
        labels,
        save_path='optimizer_comparison.png'
    )
    
    print("\n✅ 对比图已保存: optimizer_comparison.png")
    
    # ==================== 2. 生成单独训练曲线 ====================
    print("\n" + "="*60)
    print("[2/5] 生成各优化器的训练曲线...")
    print("="*60)
    for dir_name in sorted(checkpoint_dirs):
        best_model_path = os.path.join(dir_name, 'best_model.pth')
        if os.path.exists(best_model_path):
            optimizer_name = dir_name.replace('checkpoints_', '')
            plot_dir = os.path.join(dir_name, 'plots')
            print(f"  {optimizer_name.upper()}: {plot_dir}/training_history.png")
            load_and_plot_checkpoint(best_model_path, save_dir=plot_dir)
    
    # ==================== 3. 生成学习率曲线（每个优化器） ====================
    if include_lr_schedule and checkpoint_paths:
        print("\n" + "="*60)
        print("[3/6] 生成各优化器的学习率曲线...")
        print("="*60)
        
        for path, label in zip(checkpoint_paths, labels):
            optimizer_name = label.split()[0].lower()
            checkpoint_dir = os.path.dirname(path)
            plot_dir = os.path.join(checkpoint_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            try:
                checkpoint = torch.load(path, map_location='cpu')
                if 'learning_rates' in checkpoint:
                    learning_rates = checkpoint['learning_rates']
                    epochs_completed = len(learning_rates)
                    
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(range(1, epochs_completed + 1), learning_rates, linewidth=2)
                    ax.set_xlabel('Epoch', fontsize=12)
                    ax.set_ylabel('Learning Rate', fontsize=12)
                    ax.set_title(f'Learning Rate Schedule - {label}', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    lr_path = os.path.join(plot_dir, 'lr_schedule.png')
                    plt.savefig(lr_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  {label}: {lr_path}")
                else:
                    print(f"  {label}: ⚠️  checkpoint中没有learning_rates数据（需要重新训练）")
            except Exception as e:
                print(f"  {label}: ⚠️  生成失败: {e}")
    
    # ==================== 4. 生成混淆矩阵 ====================
    if include_confusion:
        print("\n" + "="*60)
        print("[4/6] 生成混淆矩阵...")
        print("="*60)
        
        try:
            # 加载测试集
            print("  加载测试数据...")
            data_loader = CIFAR10DataLoader(
                data_dir='./data',
                batch_size=128,
                num_workers=4,
                use_cutout=False
            )
            test_loader = data_loader.get_test_loader()
            
            # 定义类别名称
            classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 为每个优化器生成混淆矩阵
            for dir_name in sorted(checkpoint_dirs):
                best_model_path = os.path.join(dir_name, 'best_model.pth')
                if os.path.exists(best_model_path):
                    optimizer_name = dir_name.replace('checkpoints_', '')
                    plot_dir = os.path.join(dir_name, 'plots')
                    confusion_path = os.path.join(plot_dir, 'confusion_matrix.png')
                    
                    print(f"  {optimizer_name.upper()}: 生成混淆矩阵...")
                    
                    # 加载模型
                    model = create_model('wideresnet28_10', num_classes=10, dropout_rate=0.0, device=device)
                    checkpoint = torch.load(best_model_path, map_location=device)
                    
                    # 处理DataParallel保存的模型
                    state_dict = checkpoint['model_state_dict']
                    if list(state_dict.keys())[0].startswith('module.'):
                        from collections import OrderedDict
                        new_state_dict = OrderedDict()
                        for k, v in state_dict.items():
                            new_state_dict[k[7:]] = v
                        state_dict = new_state_dict
                    
                    model.load_state_dict(state_dict)
                    
                    # 生成混淆矩阵
                    plot_confusion_matrix(
                        model, test_loader, classes,
                        device=device,
                        save_path=confusion_path
                    )
                    print(f"    ✅ {confusion_path}")
            
            print("✅ 所有混淆矩阵已生成")
        except Exception as e:
            print(f"⚠️  生成混淆矩阵失败: {e}")
            print("   跳过混淆矩阵生成")
    
    # ==================== 5. 生成学习率调度曲线示例和数据增强可视化 ====================
    if include_extra:
        print("\n" + "="*60)
        print("[5/6] 生成额外可视化（示例图）...")
        print("="*60)
        
        # 绘制四个优化器的学习率曲线对比图
        if include_lr_schedule:
            try:
                print("  绘制四个优化器学习率对比曲线...")
                import matplotlib.pyplot as plt
                
                # 收集所有优化器的学习率数据
                lr_data = {}
                for path, label in zip(checkpoint_paths, labels):
                    try:
                        checkpoint = torch.load(path, map_location='cpu')
                        if 'learning_rates' in checkpoint:
                            lr_data[label] = checkpoint['learning_rates']
                            print(f"    ✓ {label}: 找到 {len(checkpoint['learning_rates'])} 个epoch数据")
                        else:
                            print(f"    ✗ {label}: 无learning_rates")
                    except Exception as e:
                        print(f"    ✗ {label}: 读取失败 - {str(e)[:50]}")
                
                if len(lr_data) > 0:
                    # 输出学习率统计信息
                    print("\n  学习率数据统计:")
                    for label, lrs in lr_data.items():
                        print(f"    {label}: 初始={lrs[0]:.6f}, 最终={lrs[-1]:.6e}, 最大={max(lrs):.6f}, 最小={min(lrs):.6e}")
                    
                    # 在一个图表里绘制所有学习率曲线
                    fig, ax = plt.subplots(figsize=(12, 7))
                    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']  # 红蓝绿橙
                    linestyles = ['-', '--', '-.', ':']  # 不同线型以区分重叠曲线
                    markers = ['o', 's', '^', 'D']  # 不同标记点
                    
                    for idx, (label, learning_rates) in enumerate(lr_data.items()):
                        epochs = range(1, len(learning_rates) + 1)
                        color = colors[idx % len(colors)]
                        linestyle = linestyles[idx % len(linestyles)]
                        marker = markers[idx % len(markers)]
                        
                        # 每隔20个epoch显示一个标记点，避免图表过于拥挤
                        markevery = max(1, len(learning_rates) // 10)
                        
                        ax.plot(epochs, learning_rates, linewidth=3.0, 
                               label=label, color=color, alpha=0.9, 
                               linestyle=linestyle, marker=marker, 
                               markersize=8, markevery=markevery, markerfacecolor='white',
                               markeredgewidth=2, markeredgecolor=color)
                    
                    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
                    ax.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
                    ax.set_title('Learning Rate Schedule Comparison', 
                                fontsize=15, fontweight='bold', pad=15)
                    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.set_yscale('log')
                    plt.tight_layout()
                    
                    plt.savefig('lr_schedule_cosine.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"    ✅ lr_schedule_cosine.png (包含 {len(lr_data)} 条曲线)")
                else:
                    print(f"    ⚠️  学习率数据不足4个优化器 (找到{len(lr_data)}个)")
            except Exception as e:
                print(f"    ⚠️  绘制学习率对比图失败: {e}")
        
        # 数据增强可视化
        if include_augmentation:
            try:
                print("  生成数据增强可视化...")
                data_loader = CIFAR10DataLoader(
                    data_dir='./data',
                    batch_size=128,
                    num_workers=4,
                    use_cutout=True
                )
                train_loader, _ = data_loader.get_train_valid_loader()
                
                visualize_augmentations(
                    train_loader.dataset,
                    num_samples=8,
                    save_path='data_augmentation.png'
                )
                print("    ✅ data_augmentation.png")
            except Exception as e:
                print(f"    ⚠️  生成数据增强可视化失败: {e}")
    
    # ==================== 6. 生成Optuna调参结果可视化 ====================
    if include_optuna:
        print("\n" + "="*60)
        print("[6/6] 生成Optuna调参结果可视化...")
        print("="*60)
        
        try:
            from optuna_tools.optuna_visualize import OptunaVisualizer
            
            # 查找所有optuna结果目录
            optuna_results_dir = 'optuna_results'
            if os.path.exists(optuna_results_dir):
                optuna_dirs = glob.glob(os.path.join(optuna_results_dir, '*_optuna_*'))
                
                if optuna_dirs:
                    print(f"  找到 {len(optuna_dirs)} 个Optuna调参结果")
                    
                    for optuna_dir in sorted(optuna_dirs):
                        optimizer_name = os.path.basename(optuna_dir).split('_optuna_')[0].upper()
                        print(f"\n  {optimizer_name}: 生成可视化...")
                        
                        try:
                            visualizer = OptunaVisualizer(optuna_dir)
                            visualizer.generate_report()
                            
                            viz_dir = os.path.join(optuna_dir, 'visualizations')
                            print(f"    ✅ {viz_dir}/optimization_history.png")
                            print(f"    ✅ {viz_dir}/param_importance.png")
                            print(f"    ✅ {viz_dir}/param_distributions.png")
                        except Exception as e:
                            print(f"    ⚠️  生成失败: {e}")
                    
                    print("\n✅ Optuna可视化已生成")
                else:
                    print("  未找到Optuna调参结果")
            else:
                print("  未找到optuna_results目录")
        except ImportError:
            print("  ⚠️  无法导入optuna_visualize模块")
        except Exception as e:
            print(f"  ⚠️  生成Optuna可视化失败: {e}")
    
    # ==================== 总结 ====================
    print("\n" + "="*60)
    print("所有图表生成完成！")
    print("="*60)
    print("\n📊 生成的图表:")
    print("\n  全局对比图:")
    print("    • optimizer_comparison.png - 优化器性能对比")
    if include_lr_schedule:
        print("    • lr_schedule_cosine.png - 学习率调度曲线（4个优化器对比）")
    if include_augmentation:
        print("    • data_augmentation.png - 数据增强效果")
    
    print("\n  各优化器训练图表:")
    for dir_name in sorted(checkpoint_dirs):
        optimizer_name = dir_name.replace('checkpoints_', '').upper()
        plot_dir = os.path.join(dir_name, 'plots')
        print(f"\n  {optimizer_name}:")
        print(f"    • {plot_dir}/training_history.png - 训练曲线")
        if include_confusion:
            confusion_path = os.path.join(plot_dir, 'confusion_matrix.png')
            if os.path.exists(confusion_path):
                print(f"    • {plot_dir}/confusion_matrix.png - 混淆矩阵")
    
    # 显示Optuna结果
    if include_optuna:
        optuna_results_dir = 'optuna_results'
        if os.path.exists(optuna_results_dir):
            optuna_dirs = glob.glob(os.path.join(optuna_results_dir, '*_optuna_*'))
            if optuna_dirs:
                print("\n  Optuna调参结果:")
                for optuna_dir in sorted(optuna_dirs):
                    optimizer_name = os.path.basename(optuna_dir).split('_optuna_')[0].upper()
                    viz_dir = os.path.join(optuna_dir, 'visualizations')
                    if os.path.exists(viz_dir):
                        print(f"\n  {optimizer_name} (Optuna):")
                        print(f"    • {viz_dir}/optimization_history.png")
                        print(f"    • {viz_dir}/param_importance.png")
                        print(f"    • {viz_dir}/param_distributions.png")
                        print(f"    • {viz_dir}/summary.txt")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成所有训练结果可视化')
    parser.add_argument('--no-confusion', action='store_true',
                       help='不生成混淆矩阵（跳过加载模型和测试集）')
    parser.add_argument('--no-extra', action='store_true',
                       help='只生成基础图表（训练曲线和对比图）')
    parser.add_argument('--no-lr', action='store_true',
                       help='不生成学习率调度曲线')
    parser.add_argument('--no-aug', action='store_true',
                       help='不生成数据增强可视化')
    parser.add_argument('--no-optuna', action='store_true',
                       help='不生成Optuna调参结果可视化')
    
    args = parser.parse_args()
    
    plot_all_results(
        include_extra=not args.no_extra,
        include_confusion=not args.no_confusion,
        include_lr_schedule=not args.no_lr,
        include_augmentation=not args.no_aug,
        include_optuna=not args.no_optuna
    )
