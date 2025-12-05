"""
CIFAR-10分类主程序
目标: 达到97%以上的测试准确率
"""
import argparse
import random
import numpy as np
import torch
from utils.ops_ev import run_all_optimizers, run_single_training

# 支持的优化器列表
SUPPORTED_OPTIMIZERS = ['sgd', 'adam', 'adamw', 'rmsprop']


if __name__ == '__main__':
    # ==================== 参数配置 ====================
    parser = argparse.ArgumentParser(description='CIFAR-10')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='wideresnet28_10',
                        choices=['wideresnet28_10', 'wideresnet40_10'],
                        help='模型架构选择')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=SUPPORTED_OPTIMIZERS + ['all'],
                        help='优化器选择（使用all可一次性运行所有优化器）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='权重衰减')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'multistep'],
                        help='学习率调度器')
    
    # 数据增强参数
    parser.add_argument('--use_mixup', action='store_true', default=True,
                        help='使用Mixup')
    parser.add_argument('--mixup_alpha', type=float, default=0.3,
                        help='Mixup alpha')
    parser.add_argument('--use_cutout', action='store_true', default=True,
                        help='使用Cutout')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='标签平滑')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数（2张卡时设置为8，4张卡16）')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='验证集比例')
    
    # GPU参数
    parser.add_argument('--multi_gpu', action='store_true', default=True,
                        help='使用多GPU')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='GPU IDs')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='保存目录')
    parser.add_argument('--seed', type=int, default=1009,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['train', 'test', 'both'],
                        help='运行模式')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练路径')
    parser.add_argument('--use_optuna', action='store_true', default=False,
                        help='使用Optuna最佳参数（默认不使用）')
    
    args = parser.parse_args()
    
    # ==================== 设置随机种子 ====================
    # ops_al.py中 set_seed(seed=1009): 要同步
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # ==================== 主流程 ====================
    if args.optimizer == 'all':
        run_all_optimizers(args, optimizers=SUPPORTED_OPTIMIZERS)
    else:
        run_single_training(args)
