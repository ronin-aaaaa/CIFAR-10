"""
Optuna调参最佳参数配置
根据optuna_results目录中的调参结果整理
使用方法: 在run_results.py中直接使用这些参数

quick：10次调参，50轮训练
"""

# ====== SGD优化器最佳参数 (准确率: 95.98%) ======
SGD_BEST_PARAMS = {
    'learning_rate': 0.10020323288654791,
    'weight_decay': 0.0006742324629443789,
    'mixup_alpha': 0.23965535919396147,
    'momentum': 0.8544757952585526,
    'label_smoothing': 0.14879629348122808
}

# ====== Adam优化器最佳参数 (准确率: 93.58%) ======
ADAM_BEST_PARAMS = {
    'learning_rate': 0.0008095668753633523,
    'weight_decay': 1.903738758074735e-05,
    'mixup_alpha': 0.7106939074148321,
    'beta1': 0.8773448898406806,
    'beta2': 0.9978717535606759,
    'label_smoothing': 0.10482422450882216
}

# ====== AdamW优化器最佳参数 (准确率: 93.06%) ======
ADAMW_BEST_PARAMS = {
    'learning_rate': 0.000334623042423416,
    'weight_decay': 0.00043116019157515167,
    'mixup_alpha': 0.8967395398804704,
    'beta1': 0.8983198506231269,
    'beta2': 0.9988580135532833,
    'label_smoothing': 0.15915632677328184
}

# ====== RMSprop优化器最佳参数 (准确率: 93.67%) ======
RMSPROP_BEST_PARAMS = {
    'learning_rate': 0.000101624146267342,
    'weight_decay': 2.3736705093416045e-05,
    'mixup_alpha': 0.4747327170792728,
    'alpha': 0.9626450003943534,
    'momentum': 0.7459286931890298,
    'label_smoothing': 0.05991383979849037
}

# ====== 获取最佳参数的辅助函数 ======
def get_best_params(optimizer_name):
    """
    根据优化器名称获取最佳参数
    
    参数:
        optimizer_name: 优化器名称 ('sgd', 'adam', 'adamw', 'rmsprop')
    
    返回:
        最佳参数字典
    """
    params_map = {
        'sgd': SGD_BEST_PARAMS,
        'adam': ADAM_BEST_PARAMS,
        'adamw': ADAMW_BEST_PARAMS,
        'rmsprop': RMSPROP_BEST_PARAMS
    }
    
    return params_map.get(optimizer_name.lower(), None)


def print_best_params_summary():
    """打印所有优化器的最佳参数总结"""
    print("\n" + "="*80)
    print("Optuna调参最佳参数总结")
    print("="*80)
    
    print("\n🥇 SGD (最佳准确率: 95.98%)")
    print("-" * 80)
    for key, value in SGD_BEST_PARAMS.items():
        print(f"  {key:20s}: {value}")
    
    print("\n🥈 RMSprop (准确率: 93.67%)")
    print("-" * 80)
    for key, value in RMSPROP_BEST_PARAMS.items():
        print(f"  {key:20s}: {value}")
    
    print("\n🥉 Adam (准确率: 93.58%)")
    print("-" * 80)
    for key, value in ADAM_BEST_PARAMS.items():
        print(f"  {key:20s}: {value}")
    
    print("\n   AdamW (准确率: 93.06%)")
    print("-" * 80)
    for key, value in ADAMW_BEST_PARAMS.items():
        print(f"  {key:20s}: {value}")
    
    print("\n" + "="*80)
    print("💡 建议: SGD优化器表现最佳，推荐使用SGD进行正式训练")
    print("="*80 + "\n")


if __name__ == '__main__':
    # 直接运行此文件可以查看所有最佳参数
    print_best_params_summary()
