"""
算法类代码
包含优化器参数管理、模型创建等算法相关功能
"""
import torch
import random
import numpy as np
from models.wideresnet import wideresnet28_10, wideresnet40_10

# ==================== Optuna参数管理 ====================

def _load_optuna_params():
    """内部函数：加载Optuna最佳参数"""
    try:
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from optuna_best_params import get_best_params
        return get_best_params
    except ImportError:
        return None


def get_optimizer_lr(optimizer_name, custom_lr=None, use_optuna=True):
    """
    根据优化器类型返回最佳学习率
    优先使用Optuna调优结果，否则使用默认值
    """
    if custom_lr is not None:
        return custom_lr
    
    if use_optuna:
        get_best_params = _load_optuna_params()
        if get_best_params:
            best_params = get_best_params(optimizer_name)
            if best_params and 'learning_rate' in best_params:
                return best_params['learning_rate']
    
    lr_map = {
        'sgd': 0.1,
        'adam': 0.001,
        'adamw': 0.001,
        'rmsprop': 0.001
    }
    
    return lr_map.get(optimizer_name.lower(), 0.1)


def get_optimizer_weight_decay(optimizer_name, custom_wd=None, use_optuna=True):
    """
    根据优化器类型返回最佳权重衰减
    优先使用Optuna调优结果，否则使用默认值
    """
    if custom_wd is not None:
        return custom_wd
    
    if use_optuna:
        get_best_params = _load_optuna_params()
        if get_best_params:
            best_params = get_best_params(optimizer_name)
            if best_params and 'weight_decay' in best_params:
                return best_params['weight_decay']
    
    wd_map = {
        'sgd': 5e-4,
        'adam': 1e-4,
        'adamw': 5e-4,
        'rmsprop': 1e-4
    }
    
    return wd_map.get(optimizer_name.lower(), 5e-4)


def get_optimizer_mixup_alpha(optimizer_name, custom_alpha=None, use_optuna=True):
    """
    根据优化器类型返回最佳Mixup Alpha
    优先使用Optuna调优结果，否则使用默认值
    """
    if custom_alpha is not None:
        return custom_alpha
    
    if use_optuna:
        get_best_params = _load_optuna_params()
        if get_best_params:
            best_params = get_best_params(optimizer_name)
            if best_params and 'mixup_alpha' in best_params:
                return best_params['mixup_alpha']
    
    return 0.3


def get_optimizer_label_smoothing(optimizer_name, custom_ls=None, use_optuna=True):
    """
    根据优化器类型返回最佳Label Smoothing
    优先使用Optuna调优结果，否则使用默认值
    """
    if custom_ls is not None:
        return custom_ls
    
    if use_optuna:
        get_best_params = _load_optuna_params()
        if get_best_params:
            best_params = get_best_params(optimizer_name)
            if best_params and 'label_smoothing' in best_params:
                return best_params['label_smoothing']
    
    return 0.1


# ==================== 模型创建和初始化 ====================

def set_seed(seed=1009):
    """设置随机种子以保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def create_model(model_name, num_classes=10, dropout_rate=0.0, device='cuda'):
    """
    创建模型
    
    参数:
        model_name: 模型名称
        num_classes: 分类数量
        dropout_rate: Dropout率
        device: 运行设备
    
    返回:
        model: 创建的模型
    """
    print(f"\n创建模型: {model_name}")
    
    if model_name == 'wideresnet28_10':
        model = wideresnet28_10(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'wideresnet40_10':
        model = wideresnet40_10(num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    return model
