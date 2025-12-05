"""
评估类代码
包含训练、测试、批量运行和工作流程管理
"""
import torch
import os
import time
import copy
import json
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from utils.ops_io import CIFAR10DataLoader
from utils.ops_tt import Trainer, Tester
from utils.ops_al import set_seed, create_model, get_optimizer_lr, get_optimizer_weight_decay, get_optimizer_mixup_alpha, get_optimizer_label_smoothing

TZ_CN = timezone(timedelta(hours=8))


def print_summary(results, total_duration, args):
    """
    打印实验总结报告
    
    参数:
        results: 结果列表
        total_duration: 总用时（秒）
        args: 命令行参数
    """
    print("\n" + "="*70)
    print("实验完成!")
    print("="*70)
    print(f"总用时: {total_duration/3600:.2f} 小时")
    print(f"结束时间: {datetime.now(TZ_CN).strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n各优化器结果:")
    print("-" * 70)
    
    for result in results:
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"{result['optimizer'].upper():10s} | {status:8s} | 用时: {result['duration_minutes']:6.1f}分钟", end="")
        
        if result['success']:
            if result['best_acc'] is not None:
                print(f" | 验证: {result['best_acc']:5.2f}%", end="")
            if result['test_acc'] is not None:
                print(f" | 测试: {result['test_acc']:5.2f}%", end="")
            print(f" | 目录: {result['save_dir']}")
        else:
            print(f" | 错误: {result['error']}")
    
    print("-" * 70)
    
    # 保存结果到JSON
    result_file = "optimizer_comparison.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'start_time': datetime.now(TZ_CN).strftime('%Y-%m-%d %H:%M:%S'),
            'total_duration_hours': total_duration / 3600,
            'config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'model': args.model,
                'gpu_ids': args.gpu_ids,
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存至: {result_file}")
    
    # 保存txt格式报告
    txt_file = result_file.replace('.json', '.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("CIFAR-10 优化器对比实验报告\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"开始时间: {datetime.now(TZ_CN).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总用时: {total_duration/3600:.2f} 小时\n\n")
        
        f.write("实验配置:\n")
        f.write(f"  训练轮数: {args.epochs}\n")
        f.write(f"  批次大小: {args.batch_size}\n")
        f.write(f"  模型: {args.model}\n")
        f.write(f"  GPU: {args.gpu_ids}\n\n")
        
        f.write("-"*70 + "\n")
        header = "  优化器   |   状态   |      用时      |   验证准确率   |   测试准确率\n"
        f.write(header)
        f.write("-"*70 + "\n")
        
        for result in results:
            status = "✅ 成功" if result['success'] else "❌ 失败"
            optimizer = result['optimizer'].upper()
            duration = f"{result['duration_minutes']:.1f}分钟"
            
            if result['success']:
                best_acc = f"{result['best_acc']:.2f}%" if result['best_acc'] else "N/A"
                test_acc = f"{result['test_acc']:.2f}%" if result['test_acc'] else "N/A"
                f.write(f"{optimizer:^10} | {status:^8} | {duration:^15} | {best_acc:^12} | {test_acc:^12}\n")
            else:
                f.write(f"{optimizer:^10} | {status:^8} | {duration:^15} | N/A          | N/A\n")
                f.write(f"  错误: {result['error']}\n")
        
        f.write("-"*70 + "\n")
        
        # 最佳优化器
        successful_results = [r for r in results if r['success']]
        if successful_results:
            if args.mode in ['both', 'test'] and any(r['test_acc'] for r in successful_results):
                best_result = max([r for r in successful_results if r['test_acc']], 
                                key=lambda x: x['test_acc'])
                f.write(f"\n🏆 最佳优化器 (测试集): {best_result['optimizer'].upper()} - {best_result['test_acc']:.2f}%\n")
            elif args.mode in ['both', 'train'] and any(r['best_acc'] for r in successful_results):
                best_result = max([r for r in successful_results if r['best_acc']], 
                                key=lambda x: x['best_acc'])
                f.write(f"\n🏆 最佳优化器 (验证集): {best_result['optimizer'].upper()} - {best_result['best_acc']:.2f}%\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("实验完成!\n")
    
    print(f"TXT报告已保存至: {txt_file}")
    
    # 找出最佳优化器
    successful_results = [r for r in results if r['success']]
    if successful_results:
        if args.mode in ['both', 'test'] and any(r['test_acc'] for r in successful_results):
            best_result = max([r for r in successful_results if r['test_acc']], 
                            key=lambda x: x['test_acc'])
            print(f"\n🏆 最佳优化器 (测试集): {best_result['optimizer'].upper()} - {best_result['test_acc']:.2f}%")
        elif args.mode in ['both', 'train'] and any(r['best_acc'] for r in successful_results):
            best_result = max([r for r in successful_results if r['best_acc']], 
                            key=lambda x: x['best_acc'])
            print(f"\n🏆 最佳优化器 (验证集): {best_result['optimizer'].upper()} - {best_result['best_acc']:.2f}%")
    
    print("\n🎉 所有优化器实验完成!")
    
    # 自动生成对比图表
    print("\n" + "="*70)
    print("正在生成对比图表...")
    print("="*70)
    try:
        from utils.ops_viz import compare_models
        
        checkpoint_paths = []
        plot_labels = []
        for result in successful_results:
            checkpoint_path = f"{result['save_dir']}/best_model.pth"
            if os.path.exists(checkpoint_path):
                checkpoint_paths.append(checkpoint_path)
                plot_labels.append(result['optimizer'].upper())
        
        if checkpoint_paths:
            compare_models(checkpoint_paths, plot_labels, 'optimizer_comparison.png')
            print("\n✅ 对比图已保存: optimizer_comparison.png")
    except Exception as e:
        print(f"\n⚠️  生成对比图失败: {e}")
        print("   可手动运行: python plot_results.py")


def train_model(args):
    """训练模型"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 多GPU配置
    use_multi_gpu = False
    gpu_ids = None
    
    if args.multi_gpu and torch.cuda.is_available():
        # 解析GPU IDs
        if args.gpu_ids:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        else:
            # 自动使用所有可用GPU
            gpu_ids = list(range(torch.cuda.device_count()))
        
        if len(gpu_ids) > 1:
            use_multi_gpu = True
            # 使用第一个指定的GPU作为主设备
            device = torch.device(f'cuda:{gpu_ids[0]}')
            print(f"\n多GPU训练模式")
            print(f"检测到 {torch.cuda.device_count()} 个可用GPU")
            print(f"使用GPU: {gpu_ids}")
            print(f"主设备: cuda:{gpu_ids[0]}")
            for gpu_id in gpu_ids:
                print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            # 计算总批次大小（使用局部变量，不修改args）
            per_gpu_batch_size = args.batch_size
            total_batch_size = per_gpu_batch_size * len(gpu_ids)
            print(f"总批次大小: {total_batch_size} ({per_gpu_batch_size} × {len(gpu_ids)} GPUs)")
        else:
            device = torch.device(f'cuda:{gpu_ids[0]}' if gpu_ids else 'cuda')
            print(f"\n只有一个GPU可用，使用单GPU模式")
            print(f"使用设备: {device}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f"\n使用设备: {device}")
    
    # 创建数据加载器
    print("\n加载数据集...")
    data_loader = CIFAR10DataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cutout=args.use_cutout,
        validation_split=args.validation_split
    )
    
    train_loader, valid_loader = data_loader.get_train_valid_loader()
    print(f"训练集大小: {len(train_loader.sampler)}")
    print(f"验证集大小: {len(valid_loader.sampler)}")
    
    # 创建模型
    model = create_model(args.model, num_classes=10, dropout_rate=args.dropout, device=device)
    
    # 如果启用多GPU，包装为DataParallel
    if use_multi_gpu:
        print(f"\n使用DataParallel包装模型...")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print(f"模型已分布到 {len(gpu_ids)} 个GPU上")
    # 自动获取优化器对应的最佳学习率和权重衰减
    use_optuna = getattr(args, 'use_optuna', True)
    learning_rate = get_optimizer_lr(args.optimizer, args.lr, use_optuna)
    weight_decay = get_optimizer_weight_decay(args.optimizer, args.weight_decay, use_optuna)
    
    print(f"\n优化器: {args.optimizer.upper()}")
    print(f"学习率: {learning_rate}" + (" (自动配置)" if args.lr is None else " (用户指定)"))
    print(f"权重衰减: {weight_decay}" + (" (自动配置)" if args.weight_decay is None else " (用户指定)"))
    
    use_optuna_mixup = args.lr is None and args.weight_decay is None
    if use_optuna_mixup and use_optuna:
        mixup_alpha = get_optimizer_mixup_alpha(args.optimizer, None, use_optuna)
        label_smoothing = get_optimizer_label_smoothing(args.optimizer, None, use_optuna)
    else:
        mixup_alpha = args.mixup_alpha
        label_smoothing = args.label_smoothing
        
    # 创建日志文件
    log_file = os.path.join(args.save_dir, 'training_log.txt')
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        optimizer_name=args.optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=args.epochs,
        use_mixup=args.use_mixup,
        mixup_alpha=mixup_alpha,
        label_smoothing=label_smoothing,
        scheduler_type=args.scheduler,
        log_file=log_file
    )
    
    # 如果指定了恢复路径，加载检查点
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\n从检查点恢复训练: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.best_acc = checkpoint.get('best_acc', 0.0)
            print(f"已加载检查点 (最佳准确率: {trainer.best_acc:.2f}%)")
    # 开始训练
    best_acc = trainer.train(save_path=args.save_dir)
    return best_acc


def test_model(args):
    """测试模型"""
    # 多GPU配置
    use_multi_gpu = False
    gpu_ids = None
    
    if args.multi_gpu and torch.cuda.is_available():
        if args.gpu_ids:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
        
        if len(gpu_ids) > 1:
            use_multi_gpu = True
            # 使用第一个指定的GPU作为主设备
            device = torch.device(f'cuda:{gpu_ids[0]}')
            print(f"\n多GPU测试模式")
            print(f"使用GPU: {gpu_ids}")
            print(f"主设备: cuda:{gpu_ids[0]}")
        else:
            device = torch.device(f'cuda:{gpu_ids[0]}' if gpu_ids else 'cuda')
            print(f"\n使用设备: {device}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f"\n使用设备: {device}")
    
    # 创建数据加载器
    print("\n加载测试集...")
    data_loader = CIFAR10DataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cutout=False  # 测试时不使用数据增强
    )
    
    test_loader = data_loader.get_test_loader()
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    model = create_model(args.model, num_classes=10, dropout_rate=args.dropout, device=device)
    
    # 加载最佳模型
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.isfile(best_model_path):
        print(f"\n加载最佳模型: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        
        # 处理DataParallel保存的模型
        state_dict = checkpoint['model_state_dict']
        # 如果是DataParallel保存的，需要先加载到原始模型
        if list(state_dict.keys())[0].startswith('module.'):
            # 移除'module.'前缀
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # 移除'module.'
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        print(f"模型在验证集上的最佳准确率: {checkpoint['best_acc']:.2f}%")
    else:
        print(f"\n警告: 未找到模型文件 {best_model_path}")
        print("将使用随机初始化的模型进行测试")
    
    # 如果启用多GPU，包装为DataParallel
    if use_multi_gpu:
        print(f"\n使用DataParallel包装模型...")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print(f"模型已分布到 {len(gpu_ids)} 个GPU上")
    
    # 创建测试器
    tester = Tester(model=model, test_loader=test_loader, device=device)
    
    # 测试模型（并保存测试结果到training_summary.txt）
    test_acc = tester.test(save_path=args.save_dir)
    return test_acc


def run_single_optimizer(optimizer, args, get_optimizer_lr_func, train_func, test_func):
    """运行单个优化器的训练和测试"""
    # 创建优化器专属的保存目录
    optimizer_save_dir = f"{args.save_dir.rstrip('/')}_{optimizer}"

    # 复制参数并修改优化器和保存目录
    optimizer_args = copy.deepcopy(args)
    optimizer_args.optimizer = optimizer
    optimizer_args.save_dir = optimizer_save_dir

    # 创建保存目录
    os.makedirs(optimizer_save_dir, exist_ok=True)

    # 显示配置
    print(f"\n优化器: {optimizer.upper()}")
    print(f"保存目录: {optimizer_save_dir}")
    learning_rate = get_optimizer_lr_func(optimizer, args.lr)
    print(f"学习率: {learning_rate} (自动配置)")
    print()
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行训练
        if args.mode == 'train':
            best_acc = train_func(optimizer_args)
            test_acc = None
        elif args.mode == 'test':
            test_acc = test_func(optimizer_args)
            best_acc = None
        else:  # both
            best_acc = train_func(optimizer_args)
            test_acc = test_func(optimizer_args)
        
        duration = time.time() - start_time
        success = True
        error_msg = None
        
        print(f"\n✅ {optimizer.upper()} 完成! 用时: {duration/60:.1f} 分钟")
        if best_acc is not None:
            print(f"   验证集准确率: {best_acc:.2f}%")
        if test_acc is not None:
            print(f"   测试集准确率: {test_acc:.2f}%")
            
    except Exception as e:
        duration = time.time() - start_time
        success = False
        error_msg = str(e)
        best_acc = None
        test_acc = None
        print(f"\n❌ {optimizer.upper()} 训练失败: {error_msg}")
    
    return {
        'optimizer': optimizer,
        'success': success,
        'duration_minutes': duration / 60,
        'best_acc': best_acc,
        'test_acc': test_acc,
        'save_dir': optimizer_save_dir,
        'error': error_msg
    }


def run_all_optimizers_batch(args, get_optimizer_lr_func, train_func, test_func, optimizers=None):
    """批量运行所有支持的优化器"""
    if optimizers is None:
        optimizers = ['sgd', 'adam', 'adamw', 'rmsprop']  # 默认优化器列表
    
    print("\n" + "="*70)
    print("CIFAR-10 优化器对比实验")
    print("="*70)
    print(f"开始时间: {datetime.now(TZ_CN).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n将依次训练以下优化器: {', '.join([opt.upper() for opt in optimizers])}")
    print(f"每个优化器训练 {args.epochs} 轮")
    print(f"预计总时长: {len(optimizers) * args.epochs / 200 * 1:.1f}-{len(optimizers) * args.epochs / 200 * 1.5:.1f} 小时")
    print("="*70)

    # 创建数据目录
    os.makedirs(args.data_dir, exist_ok=True)
    results = []
    total_start_time = time.time()
    
    # 依次运行每个优化器
    for i, optimizer in enumerate(optimizers, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(optimizers)}] 开始训练: {optimizer.upper()}")
        print(f"{'='*70}")
        result = run_single_optimizer(optimizer, args, get_optimizer_lr_func, train_func, test_func)
        results.append(result)
        
        # 显示进度
        completed = sum(1 for r in results if r['success'])
        print(f"\n进度: {len(results)}/{len(optimizers)} 完成, {completed} 成功")
    
    total_duration = time.time() - total_start_time
    print_summary(results, total_duration, args)


def run_all_optimizers(args, optimizers=None):
    """运行所有优化器的入口函数"""
    run_all_optimizers_batch(args, get_optimizer_lr, train_model, test_model, optimizers)


def run_single_training(args):
    """运行单个优化器训练"""
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # 打印配置信息
    print("\n" + "="*60)
    print("CIFAR-10 高精度分类训练")
    print("="*60)
    print("\n配置信息:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("="*60)
    
    # 根据模式运行
    if args.mode == 'train':
        print("\n开始训练模式...")
        best_acc = train_model(args)
        print(f"\n训练完成! 最佳验证准确率: {best_acc:.2f}%")
        
    elif args.mode == 'test':
        print("\n开始测试模式...")
        test_acc = test_model(args)
        print(f"\n测试完成! 测试准确率: {test_acc:.2f}%")
        
    elif args.mode == 'both':
        print("\n开始训练+测试模式...")
        
        # 训练
        print("\n[阶段1/2] 训练模型")
        best_acc = train_model(args)
        print(f"\n训练完成! 最佳验证准确率: {best_acc:.2f}%")
        
        # 测试
        print("\n[阶段2/2] 测试模型")
        test_acc = test_model(args)
        print(f"\n测试完成! 测试准确率: {test_acc:.2f}%")
        
        # 总结
        print("\n" + "="*60)
        print("训练和测试总结:")
        print(f"  验证集最佳准确率: {best_acc:.2f}%")
        print(f"  测试集准确率: {test_acc:.2f}%")
    
    print("\n程序运行完成!")
