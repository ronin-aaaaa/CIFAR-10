"""
Optuna自动调参系统 for CIFAR-10
支持SGD, Adam, AdamW, RMSprop四种优化器的超参数优化
"""
import torch
import argparse
import os
import sys
import optuna
from optuna.trial import TrialState
import json
from datetime import datetime, timezone, timedelta
import random
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 现在可以正常导入项目模块
from utils.ops_al import create_model, get_optimizer_lr, get_optimizer_weight_decay, set_seed
from utils.ops_io import CIFAR10DataLoader
from utils.ops_tt import Trainer, Tester



class OptunaOptimizer:
    """Optuna优化器封装类"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 使用北京时间 (UTC+08:00)
        self.tz = timezone(timedelta(hours=8))
        self.start_time = datetime.now(self.tz)
        
        # 创建数据加载器（只创建一次，节省时间）
        print(f"\n加载数据集...")
        self.data_loader = CIFAR10DataLoader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_cutout=args.use_cutout,
            validation_split=args.validation_split
        )
        self.train_loader, self.valid_loader = self.data_loader.get_train_valid_loader()
        print(f"训练集大小: {len(self.train_loader.sampler)}")
        print(f"验证集大小: {len(self.valid_loader.sampler)}")
        
        # 创建测试集加载器（用于最终测试评估）
        self.test_loader = self.data_loader.get_test_loader()
        
        # 创建保存目录
        self.study_name = f"{args.optimizer}_optuna_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = os.path.join(args.save_dir, self.study_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 保存配置
        self._save_config()
    
    def _save_config(self):
        """保存实验配置"""
        config = {
            'optimizer': self.args.optimizer,
            'model': self.args.model,
            'epochs': self.args.epochs,
            'batch_size': self.args.batch_size,
            'n_trials': self.args.n_trials,
            'device': str(self.device),
            'timestamp': self.start_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    
    def _get_param_ranges(self, trial, optimizer_name):
        """根据优化器类型获取参数搜索范围"""
        params = {}
        
        if optimizer_name == 'sgd':
            # SGD参数范围（当前96.9%，微调以达到97%+）
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.05, 0.15, log=True)
            params['weight_decay'] = trial.suggest_float('weight_decay', 1e-4, 1e-3, log=True)
            params['mixup_alpha'] = trial.suggest_float('mixup_alpha', 0.2, 0.6)
            params['momentum'] = trial.suggest_float('momentum', 0.85, 0.95)
            params['label_smoothing'] = trial.suggest_float('label_smoothing', 0.05, 0.15)
            
        elif optimizer_name == 'adamw':
            # AdamW参数范围（当前96.1%，目标97%+）
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.0001, 0.005, log=True)
            params['weight_decay'] = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)  # 关键！
            params['mixup_alpha'] = trial.suggest_float('mixup_alpha', 0.8, 1.5)  # 强增强
            params['beta1'] = trial.suggest_float('beta1', 0.85, 0.95)
            params['beta2'] = trial.suggest_float('beta2', 0.99, 0.9999)
            params['label_smoothing'] = trial.suggest_float('label_smoothing', 0.05, 0.2)
            
        elif optimizer_name == 'adam':
            # Adam参数范围（当前94.6%，目标96%+）
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.0001, 0.003, log=True)
            params['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 5e-4, log=True)
            params['mixup_alpha'] = trial.suggest_float('mixup_alpha', 0.5, 1.2)
            params['beta1'] = trial.suggest_float('beta1', 0.85, 0.95)
            params['beta2'] = trial.suggest_float('beta2', 0.99, 0.9999)
            params['label_smoothing'] = trial.suggest_float('label_smoothing', 0.05, 0.15)
            
        elif optimizer_name == 'rmsprop':
            # RMSprop参数范围（当前94.5%，目标96%+）
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.0001, 0.005, log=True)
            params['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 5e-4, log=True)
            params['mixup_alpha'] = trial.suggest_float('mixup_alpha', 0.4, 1.0)
            params['alpha'] = trial.suggest_float('alpha', 0.9, 0.999)
            params['momentum'] = trial.suggest_float('momentum', 0.0, 0.9)
            params['label_smoothing'] = trial.suggest_float('label_smoothing', 0.05, 0.15)
        
        return params
    
    def objective(self, trial):
        """Optuna的目标函数"""
        # 设置随机种子（每次trial使用不同的种子以避免过拟合）
        seed = self.args.seed + trial.number
        set_seed(seed)
        
        # 获取超参数
        params = self._get_param_ranges(trial, self.args.optimizer)
        
        # 打印当前trial的参数
        print(f"\n{'='*80}")
        print(f"Trial {trial.number + 1}/{self.args.n_trials}")
        print(f"{'='*80}")
        print("Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print(f"{'='*80}\n")
        
        # 处理GPU设备
        if hasattr(self.args, 'gpu_ids') and self.args.gpu_ids:
            gpu_ids = [int(x.strip()) for x in self.args.gpu_ids.split(',')]
            device = torch.device(f'cuda:{gpu_ids[0]}')
        else:
            device = self.device
            gpu_ids = None
        
        # 创建模型（传入正确的参数）
        dropout_rate = params.get('dropout', self.args.dropout if hasattr(self.args, 'dropout') else 0.0)
        model = create_model(
            model_name=self.args.model,
            num_classes=10,
            dropout_rate=dropout_rate,
            device=device
        )
        
        # 多GPU支持
        if self.args.multi_gpu and torch.cuda.device_count() > 1:
            if gpu_ids:
                print(f"使用指定GPU: {gpu_ids}")
                model = torch.nn.DataParallel(model, device_ids=gpu_ids)
            else:
                print(f"使用所有GPU: {torch.cuda.device_count()} 个")
                model = torch.nn.DataParallel(model)
        
        # 创建训练器（需要修改以支持动态参数）
        trainer = self._create_trainer(model, params, device)
        
        # 训练模型
        try:
            best_val_acc = trainer.train(save_path=self.save_dir)
            
            # 在测试集上评估，用测试集准确率作为optuna的优化目标
            test_acc = self._test_on_test_set(model, device)
            
            print(f"\n📊 Trial {trial.number} 结果:")
            print(f"  验证集最佳准确率: {best_val_acc:.4f}%")
            print(f"  测试集准确率: {test_acc:.4f}%")
            print(f"  验证-测试差异: {best_val_acc - test_acc:.4f}%")
            
            # 如果差异过大，发出警告
            if best_val_acc - test_acc > 2.0:  # 差异超过2%
                print(f"  ⚠️  警告: 验证集和测试集性能差异较大，可能存在过拟合!")
            
            # 保存trial结果
            trial_result = {
                'trial_number': trial.number,
                'params': params,
                'validation_accuracy': best_val_acc,
                'test_accuracy': test_acc,
                'val_test_gap': best_val_acc - test_acc,
                'seed': seed
            }
            
            result_file = os.path.join(self.save_dir, f'trial_{trial.number}.json')
            with open(result_file, 'w') as f:
                json.dump(trial_result, f, indent=4)
            
            # 🔑 关键！返回测试集准确率作为optuna的优化目标
            return test_acc
            
        except Exception as e:
            print(f"\nTrial {trial.number} 失败: {str(e)}")
            return 0.0
    
    def _create_trainer(self, model, params, device):
        """创建训练器，支持动态参数"""
        # 重用正式训练的参数设置逻辑
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            device=device,
            optimizer_name=self.args.optimizer,
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            epochs=self.args.epochs,
            use_mixup=self.args.use_mixup,
            mixup_alpha=params['mixup_alpha'],
            label_smoothing=params['label_smoothing'],
            scheduler_type=self.args.scheduler,
            log_file=None  # optuna不需要详细日志
        )
        
        # 对于有额外参数的优化器，需要重新创建优化器以支持optuna动态调参
        if self.args.optimizer == 'sgd' and 'momentum' in params:
            trainer.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=params['learning_rate'],
                momentum=params['momentum'],
                weight_decay=params['weight_decay'],
                nesterov=True
            )
        elif self.args.optimizer in ['adam', 'adamw'] and 'beta1' in params:
            optimizer_class = torch.optim.AdamW if self.args.optimizer == 'adamw' else torch.optim.Adam
            trainer.optimizer = optimizer_class(
                model.parameters(),
                lr=params['learning_rate'],
                betas=(params['beta1'], params['beta2']),
                weight_decay=params['weight_decay'],
                eps=1e-8
            )
        elif self.args.optimizer == 'rmsprop' and 'alpha' in params:
            trainer.optimizer = torch.optim.RMSprop(
                model.parameters(),
                lr=params['learning_rate'],
                alpha=params['alpha'],
                momentum=params.get('momentum', 0),
                weight_decay=params['weight_decay'],
                eps=1e-8
            )
        
        # 重新创建学习率调度器
        if self.args.scheduler == 'cosine':
            trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer, T_max=self.args.epochs, eta_min=1e-6
            )
        elif self.args.scheduler == 'multistep':
            trainer.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                trainer.optimizer, milestones=[60, 120, 160], gamma=0.2
            )
        
        return trainer
    
    def _test_on_test_set(self, model, device):
        """在测试集上评估模型性能 - 重用标准Tester类"""
        tester = Tester(model=model, test_loader=self.test_loader, device=device)
        test_acc = tester.test()
        return test_acc
    
    def optimize(self):
        """执行优化"""
        print(f"\n{'='*80}")
        print(f"开始Optuna优化 - {self.args.optimizer.upper()}")
        print(f"{'='*80}")
        print(f"优化器: {self.args.optimizer}")
        print(f"模型: {self.args.model}")
        print(f"训练轮数: {self.args.epochs}")
        print(f"总Trial数: {self.args.n_trials}")
        print(f"保存目录: {self.save_dir}")
        print(f"{'='*80}\n")
        
        # 创建Optuna study
        study = optuna.create_study(
            direction='maximize',
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=self.args.seed)
        )
        
        # 运行优化
        study.optimize(
            self.objective,
            n_trials=self.args.n_trials,
            show_progress_bar=False  # 禁用optuna进度条，避免与训练进度条冲突
        )
        
        # 保存结果
        self._save_results(study)
        
        # 打印最佳结果
        self._print_results(study)
        
        return study
    
    def _save_results(self, study):
        """保存优化结果"""
        # 保存最佳参数
        best_params = study.best_params
        best_value = study.best_value
        
        results = {
            'best_accuracy': best_value,
            'best_params': best_params,
            'best_trial_number': study.best_trial.number,
            'n_trials': len(study.trials),
            'completed_trials': len([t for t in study.trials if t.state == TrialState.COMPLETE])
        }
        
        with open(os.path.join(self.save_dir, 'best_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # 保存所有trials的历史
        trials_data = []
        for trial in study.trials:
            if trial.state == TrialState.COMPLETE:
                trials_data.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params
                })
        
        with open(os.path.join(self.save_dir, 'all_trials.json'), 'w') as f:
            json.dump(trials_data, f, indent=4)
        
        print(f"\n结果已保存到: {self.save_dir}")
    
    def _print_results(self, study):
        """打印优化结果"""
        print(f"\n{'='*80}")
        print("优化完成!")
        print(f"{'='*80}")
        print(f"最佳准确率: {study.best_value:.4f}%")
        print(f"最佳Trial: {study.best_trial.number}")
        print("\n最佳参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"{'='*80}\n")
        
        # 打印Top 5 trials
        sorted_trials = sorted(
            [t for t in study.trials if t.state == TrialState.COMPLETE],
            key=lambda t: t.value,
            reverse=True
        )[:5]
        
        print("Top 5 Trials:")
        for i, trial in enumerate(sorted_trials, 1):
            print(f"\n{i}. Trial {trial.number}: {trial.value:.4f}%")
            print("   Parameters:")
            for key, value in trial.params.items():
                print(f"     {key}: {value}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Optuna超参数优化 for CIFAR-10')
    
    # Optuna参数
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adam', 'adamw', 'rmsprop'],
                        help='要优化的优化器')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Optuna试验次数（建议至少30次）')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='wideresnet28_10',
                        choices=['wideresnet28_10', 'wideresnet40_10'],
                        help='模型架构')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                        help='每个trial的训练轮数')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'multistep'],
                        help='学习率调度器')
    
    # 数据增强参数
    parser.add_argument('--use_mixup', action='store_true', default=True,
                        help='使用Mixup数据增强')
    parser.add_argument('--use_cutout', action='store_true', default=True,
                        help='使用Cutout数据增强')
    
    # 数据参数  
    parser.add_argument('--data_dir', type=str, 
                        default=os.path.join(project_root, 'data'),
                        help='数据目录 (默认: 项目根目录/data, 可自定义绝对路径)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='验证集比例')
    
    # 多GPU参数
    parser.add_argument('--multi_gpu', action='store_true', default=True,
                        help='使用多GPU训练')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='指定使用的GPU ID，用逗号分隔，如"0,1,2,3"')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, 
                        default=os.path.join(project_root, 'optuna_results'),
                        help='结果保存目录 (默认: 项目根目录/optuna_results, 可自定义绝对路径)')
    parser.add_argument('--seed', type=int, default=1009,
                        help='随机种子')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建优化器
    optimizer = OptunaOptimizer(args)
    
    # 执行优化
    study = optimizer.optimize()
    
    print("\n优化完成! 查看结果:")
    print(f"  保存目录: {optimizer.save_dir}")
    print(f"  最佳准确率: {study.best_value:.4f}%")


if __name__ == '__main__':
    main()
