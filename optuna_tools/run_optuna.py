"""
Optuna快速启动脚本
提供便捷的命令行接口来运行超参数优化
"""
import argparse
import os
import sys

# 确保能够导入optuna_configs
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from optuna_configs import QUICK_CONFIG, FULL_CONFIG, DEEP_CONFIG, get_optimizer_config, print_optimizer_info


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Optuna超参数优化快速启动脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 快速测试AdamW优化器（10次试验，50轮训练）
  python run_optuna.py --optimizer adamw --mode quick
  
  # 完整优化SGD优化器（20次试验，200轮训练）
  python run_optuna.py --optimizer sgd --mode full
  
  # 深度优化Adam优化器（30次试验，200轮训练）
  python run_optuna.py --optimizer adam --mode deep
  
  # 一次性运行所有优化器（推荐）
  python run_optuna.py --optimizer all --mode full
  
  # 指定使用特定GPU卡（如只用卡0和卡1）
  python run_optuna.py --optimizer adamw --mode quick --gpu_ids 0,1
  
  # 查看优化器的参数搜索空间
  python run_optuna.py --optimizer adamw --info
  
  # 自定义试验次数和训练轮数
  python run_optuna.py --optimizer adamw --n_trials 30 --epochs 150
        """
    )
    
    # 必选参数
    parser.add_argument('--optimizer', type=str, required=True,
                       choices=['sgd', 'adam', 'adamw', 'rmsprop', 'all'],
                       help='要优化的优化器（使用all可一次性运行所有优化器）')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='full',
                       choices=['quick', 'full', 'deep'],
                       help='优化模式: quick(快速测试), full(完整优化), deep(深度优化)')
    
    # 信息查看
    parser.add_argument('--info', action='store_true',
                       help='只显示优化器信息，不运行优化')
    
    # 自定义参数
    parser.add_argument('--n_trials', type=int, default=None,
                       help='试验次数（覆盖mode设置）')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（覆盖mode设置）')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    
    # 其他参数
    parser.add_argument('--model', type=str, default='wideresnet28_10',
                       choices=['wideresnet28_10', 'wideresnet40_10'],
                       help='模型架构')
    parser.add_argument('--save_dir', type=str, default=os.path.join(os.path.dirname(script_dir), 'optuna_results'),
                       help='结果保存目录')
    parser.add_argument('--no_gpu', action='store_true',
                       help='禁用GPU（使用CPU训练）')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='指定使用的GPU ID，用逗号分隔，如"0,1,2,3"')
    
    args = parser.parse_args()
    
    # 如果是all，循环运行所有优化器
    if args.optimizer == 'all':
        optimizers = ['sgd', 'adam', 'adamw', 'rmsprop']
        
        # 获取配置
        if args.mode == 'quick':
            config = QUICK_CONFIG
        elif args.mode == 'full':
            config = FULL_CONFIG
        elif args.mode == 'deep':
            config = DEEP_CONFIG
        
        n_trials = args.n_trials if args.n_trials is not None else config['n_trials']
        epochs = args.epochs if args.epochs is not None else config['epochs']
        
        print(f"\n{'='*80}")
        print(f"🚀 批量运行模式：将依次运行所有优化器")
        print(f"优化器列表: {', '.join([opt.upper() for opt in optimizers])}")
        print(f"模式: {args.mode.upper()}")
        print(f"每个优化器试验次数: {n_trials}")
        print(f"每次训练轮数: {epochs}")
        print(f"{'='*80}\n")
        
        # 批量运行时只确认一次
        if args.mode in ['full', 'deep']:
            total_time_estimate = len(optimizers) * n_trials * epochs * 2 / 60
            print(f"⏱️  预计总用时: ~{total_time_estimate:.1f}小时 (4个优化器)")
            print(f"💡 提示: 如果想快速测试，使用 --mode quick")
            
            response = input("\n是否继续批量运行? [y/N]: ")
            if response.lower() != 'y':
                print("已取消")
                return
        
        for i, opt in enumerate(optimizers, 1):
            print(f"\n{'#'*80}")
            print(f"# [{i}/{len(optimizers)}] 开始优化: {opt.upper()}")
            print(f"{'#'*80}\n")
            
            # 创建临时args对象
            temp_args = argparse.Namespace(**vars(args))
            temp_args.optimizer = opt
            
            # 运行单个优化器（跳过确认）
            run_single_optimizer(temp_args, skip_confirm=True)
            
            if i < len(optimizers):
                print(f"\n{'='*80}")
                print(f"✅ {opt.upper()} 完成，准备下一个优化器...")
                print(f"{'='*80}\n")
        
        print(f"\n{'='*80}")
        print(f"🎉 所有优化器调参完成！")
        print(f"{'='*80}\n")
        return
    
    # 运行单个优化器
    run_single_optimizer(args, skip_confirm=False)


def run_single_optimizer(args, skip_confirm=False):
    """运行单个优化器的调参
    
    参数:
        args: 命令行参数
        skip_confirm: 是否跳过确认（批量运行时使用）
    """
    # 如果只是查看信息
    if args.info:
        print_optimizer_info(args.optimizer)
        return
    
    # 根据模式设置参数
    if args.mode == 'quick':
        config = QUICK_CONFIG
        print("\n🚀 快速测试模式")
    elif args.mode == 'full':
        config = FULL_CONFIG
        print("\n🎯 完整优化模式")
    elif args.mode == 'deep':
        config = DEEP_CONFIG
        print("\n🔬 深度优化模式")
    
    # 如果用户指定了自定义值，覆盖配置
    n_trials = args.n_trials if args.n_trials is not None else config['n_trials']
    epochs = args.epochs if args.epochs is not None else config['epochs']
    
    print(f"{'='*80}")
    print(f"优化器: {args.optimizer.upper()}")
    print(f"模型: {args.model}")
    print(f"试验次数: {n_trials}")
    print(f"每次训练轮数: {epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"保存目录: {args.save_dir}")
    print(f"{'='*80}\n")
    
    # 显示优化器的当前性能和目标
    opt_config = get_optimizer_config(args.optimizer)
    print(f"📊 {opt_config['description']}")
    print(f"🎯 目标: {opt_config['target']}\n")
    
    # 确认是否继续（批量运行时跳过）
    if not skip_confirm and args.mode in ['full', 'deep']:
        total_time_estimate = n_trials * epochs * 2 / 60  # 粗略估计（分钟）
        print(f"⏱️  预计总用时: ~{total_time_estimate:.1f}小时")
        print(f"💡 提示: 如果想快速测试，使用 --mode quick")
        
        response = input("\n是否继续? [y/N]: ")
        if response.lower() != 'y':
            print("已取消")
            return
    
    # 构建命令 - 使用绝对路径
    optuna_script = os.path.join(script_dir, 'optuna_tuning.py')
    cmd_parts = [
        f'python "{optuna_script}"',
        f'--optimizer {args.optimizer}',
        f'--n_trials {n_trials}',
        f'--epochs {epochs}',
        f'--batch_size {args.batch_size}',
        f'--num_workers {args.num_workers}',
        f'--model {args.model}',
        f'--save_dir {args.save_dir}'
    ]
    
    if args.no_gpu:
        cmd_parts.append('--multi_gpu False')
    
    if args.gpu_ids:
        cmd_parts.append(f'--gpu_ids {args.gpu_ids}')
    
    cmd = ' '.join(cmd_parts)
    
    print(f"\n执行命令:")
    print(f"  {cmd}\n")
    
    # 执行命令
    os.system(cmd)


if __name__ == '__main__':
    main()
