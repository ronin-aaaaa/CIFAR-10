"""
Optuna优化器参数配置
定义每个优化器的最佳搜索空间
"""

# 优化器参数搜索空间配置
OPTIMIZER_CONFIGS = {
    'sgd': {
        'description': 'SGD优化器 (当前最佳: 96.9%)',
        'target': '97.0%+',
        'params': {
            'learning_rate': {
                'range': (0.05, 0.15),
                'log': True,
                'optimal_region': (0.08, 0.12),
                'description': '学习率 - SGD通常需要较大的学习率'
            },
            'weight_decay': {
                'range': (1e-4, 1e-3),
                'log': True,
                'optimal_region': (3e-4, 7e-4),
                'description': '权重衰减 - SGD需要较大的正则化'
            },
            'mixup_alpha': {
                'range': (0.2, 0.6),
                'log': False,
                'optimal_region': (0.3, 0.5),
                'description': 'Mixup强度 - SGD泛化好，不需要太强'
            },
            'momentum': {
                'range': (0.85, 0.95),
                'log': False,
                'optimal_region': (0.88, 0.92),
                'description': 'SGD动量参数'
            },
            'label_smoothing': {
                'range': (0.05, 0.15),
                'log': False,
                'optimal_region': (0.08, 0.12),
                'description': '标签平滑系数'
            }
        }
    },
    
    'adamw': {
        'description': 'AdamW优化器 (当前最佳: 96.1%)',
        'target': '97.0%+',
        'params': {
            'learning_rate': {
                'range': (0.0001, 0.005),
                'log': True,
                'optimal_region': (0.0005, 0.003),
                'description': '学习率 - AdamW通常需要较小的学习率'
            },
            'weight_decay': {
                'range': (1e-4, 1e-2),
                'log': True,
                'optimal_region': (1e-3, 5e-3),
                'description': '权重衰减 - AdamW的核心优势！可以设置很大'
            },
            'mixup_alpha': {
                'range': (0.8, 1.5),
                'log': False,
                'optimal_region': (1.0, 1.3),
                'description': 'Mixup强度 - AdamW容易过拟合，需要强增强'
            },
            'beta1': {
                'range': (0.85, 0.95),
                'log': False,
                'optimal_region': (0.88, 0.92),
                'description': 'Adam beta1参数 - 一阶矩估计'
            },
            'beta2': {
                'range': (0.99, 0.9999),
                'log': False,
                'optimal_region': (0.995, 0.999),
                'description': 'Adam beta2参数 - 二阶矩估计'
            },
            'label_smoothing': {
                'range': (0.05, 0.2),
                'log': False,
                'optimal_region': (0.1, 0.15),
                'description': '标签平滑系数'
            }
        }
    },
    
    'adam': {
        'description': 'Adam优化器 (当前最佳: 94.6%)',
        'target': '96.0%+',
        'params': {
            'learning_rate': {
                'range': (0.0001, 0.003),
                'log': True,
                'optimal_region': (0.0003, 0.001),
                'description': '学习率 - Adam可能需要比当前更小的学习率'
            },
            'weight_decay': {
                'range': (1e-5, 5e-4),
                'log': True,
                'optimal_region': (5e-5, 2e-4),
                'description': '权重衰减 - Adam的weight_decay要小心，不能太大'
            },
            'mixup_alpha': {
                'range': (0.5, 1.2),
                'log': False,
                'optimal_region': (0.7, 1.0),
                'description': 'Mixup强度 - 中等强度'
            },
            'beta1': {
                'range': (0.85, 0.95),
                'log': False,
                'optimal_region': (0.88, 0.92),
                'description': 'Adam beta1参数'
            },
            'beta2': {
                'range': (0.99, 0.9999),
                'log': False,
                'optimal_region': (0.995, 0.999),
                'description': 'Adam beta2参数'
            },
            'label_smoothing': {
                'range': (0.05, 0.15),
                'log': False,
                'optimal_region': (0.08, 0.12),
                'description': '标签平滑系数'
            }
        }
    },
    
    'rmsprop': {
        'description': 'RMSprop优化器 (当前最佳: 94.5%)',
        'target': '96.0%+',
        'params': {
            'learning_rate': {
                'range': (0.0001, 0.005),
                'log': True,
                'optimal_region': (0.0005, 0.002),
                'description': '学习率'
            },
            'weight_decay': {
                'range': (1e-5, 5e-4),
                'log': True,
                'optimal_region': (5e-5, 2e-4),
                'description': '权重衰减'
            },
            'mixup_alpha': {
                'range': (0.4, 1.0),
                'log': False,
                'optimal_region': (0.6, 0.9),
                'description': 'Mixup强度'
            },
            'alpha': {
                'range': (0.9, 0.999),
                'log': False,
                'optimal_region': (0.95, 0.99),
                'description': 'RMSprop alpha参数 - 平滑常数'
            },
            'momentum': {
                'range': (0.0, 0.9),
                'log': False,
                'optimal_region': (0.0, 0.5),
                'description': 'RMSprop动量 - 原生不需要，可尝试添加'
            },
            'label_smoothing': {
                'range': (0.05, 0.15),
                'log': False,
                'optimal_region': (0.08, 0.12),
                'description': '标签平滑系数'
            }
        }
    }
}


# 快速配置（减少训练轮数用于快速测试）
QUICK_CONFIG = {
    'epochs': 50,  # 原本200轮，快速测试用50轮
    'n_trials': 10,  # 原本50次，快速测试用10次
    'description': '快速测试配置 - 用于验证代码和快速迭代'
}


# 完整配置（用于最终优化）
FULL_CONFIG = {
    'epochs': 200,
    'n_trials': 20,  # 降低到20次，在效果和时间之间取得平衡
    'description': '完整优化配置 - 用于获得最佳结果'
}


# 深度优化配置（用于精细调优）
DEEP_CONFIG = {
    'epochs': 200,
    'n_trials': 100,
    'description': '深度优化配置 - 更多trials以找到全局最优'
}


def get_optimizer_config(optimizer_name):
    """
    获取指定优化器的配置
    
    参数:
        optimizer_name: 优化器名称 ('sgd', 'adam', 'adamw', 'rmsprop')
    
    返回:
        配置字典
    """
    if optimizer_name not in OPTIMIZER_CONFIGS:
        raise ValueError(f"不支持的优化器: {optimizer_name}. 可选: {list(OPTIMIZER_CONFIGS.keys())}")
    
    return OPTIMIZER_CONFIGS[optimizer_name]


def print_optimizer_info(optimizer_name):
    """打印优化器的详细信息"""
    config = get_optimizer_config(optimizer_name)
    
    print(f"\n{'='*80}")
    print(f"优化器: {optimizer_name.upper()}")
    print(f"{'='*80}")
    print(f"描述: {config['description']}")
    print(f"目标: {config['target']}")
    print(f"\n参数搜索空间:")
    print(f"{'-'*80}")
    
    for param_name, param_info in config['params'].items():
        print(f"\n{param_name}:")
        print(f"  范围: {param_info['range']}")
        print(f"  最优区间: {param_info['optimal_region']}")
        print(f"  对数尺度: {param_info['log']}")
        print(f"  说明: {param_info['description']}")
    
    print(f"{'='*80}\n")


def get_default_params(optimizer_name):
    """获取优化器的默认参数（当前使用的）"""
    defaults = {
        'sgd': {
            'learning_rate': 0.1,
            'weight_decay': 5e-4,
            'mixup_alpha': 0.3,
            'momentum': 0.9,
            'label_smoothing': 0.1
        },
        'adamw': {
            'learning_rate': 0.001,
            'weight_decay': 5e-4,
            'mixup_alpha': 1.0,
            'beta1': 0.9,
            'beta2': 0.999,
            'label_smoothing': 0.1
        },
        'adam': {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'mixup_alpha': 0.3,
            'beta1': 0.9,
            'beta2': 0.999,
            'label_smoothing': 0.1
        },
        'rmsprop': {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'mixup_alpha': 0.3,
            'alpha': 0.99,
            'momentum': 0,
            'label_smoothing': 0.1
        }
    }
    
    return defaults.get(optimizer_name, {})


if __name__ == '__main__':
    """测试配置文件"""
    import sys
    
    if len(sys.argv) > 1:
        optimizer = sys.argv[1]
        print_optimizer_info(optimizer)
    else:
        print("可用的优化器:")
        for opt_name in OPTIMIZER_CONFIGS.keys():
            print(f"  - {opt_name}")
        print("\n使用方法: python optuna_configs.py <optimizer_name>")
        print("例如: python optuna_configs.py adamw")
