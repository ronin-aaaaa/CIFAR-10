"""
Optuna结果可视化工具
用于分析和可视化超参数优化结果
"""
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


class OptunaVisualizer:
    """Optuna结果可视化器"""
    
    def __init__(self, result_dir):
        """
        初始化可视化器
        
        参数:
            result_dir: Optuna结果目录路径
        """
        self.result_dir = Path(result_dir)
        self.load_results()
    
    def load_results(self):
        """加载优化结果"""
        # 加载所有trials
        all_trials_path = self.result_dir / 'all_trials.json'
        if all_trials_path.exists():
            with open(all_trials_path, 'r') as f:
                self.all_trials = json.load(f)
                self.df_trials = pd.DataFrame(self.all_trials)
        else:
            # 尝试从单独的trial_*.json文件重建数据
            print(f"警告: 未找到all_trials.json，尝试从trial_*.json文件重建...")
            self.all_trials = self._rebuild_trials_from_files()
            if self.all_trials:
                self.df_trials = pd.DataFrame(self.all_trials)
                print(f"  ✓ 成功从 {len(self.all_trials)} 个trial文件重建数据")
            else:
                print(f"  ✗ 未找到任何trial文件")
                self.df_trials = None
        
        # 加载最佳结果
        best_results_path = self.result_dir / 'best_results.json'
        if best_results_path.exists():
            with open(best_results_path, 'r') as f:
                self.best_results = json.load(f)
        else:
            # 从all_trials中重建best_results
            if self.all_trials:
                print(f"警告: 未找到best_results.json，从trial数据重建...")
                self.best_results = self._rebuild_best_results()
                if self.best_results:
                    print(f"  ✓ 重建best_results成功，最佳准确率: {self.best_results['best_accuracy']:.4f}%")
            else:
                self.best_results = None
        
        # 加载配置
        config_path = self.result_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = None
    
    def _rebuild_trials_from_files(self):
        """从单独的trial_*.json文件重建all_trials数据"""
        import glob
        trial_files = sorted(self.result_dir.glob('trial_*.json'))
        
        if not trial_files:
            return None
        
        all_trials = []
        for trial_file in trial_files:
            try:
                with open(trial_file, 'r') as f:
                    trial_data = json.load(f)
                
                # 转换为标准格式（兼容不同的字段名）
                standardized = {
                    'number': trial_data.get('trial_number', trial_data.get('number', 0)),
                    'value': trial_data.get('validation_accuracy', trial_data.get('value', 0)),
                    'params': trial_data.get('params', {}),
                    'state': 'COMPLETE'
                }
                
                # 保留其他有用字段
                if 'test_accuracy' in trial_data:
                    standardized['test_accuracy'] = trial_data['test_accuracy']
                if 'seed' in trial_data:
                    standardized['seed'] = trial_data['seed']
                
                all_trials.append(standardized)
            except Exception as e:
                print(f"  警告: 读取 {trial_file.name} 失败: {e}")
        
        # 按trial number排序
        all_trials.sort(key=lambda x: x['number'])
        return all_trials if all_trials else None
    
    def _rebuild_best_results(self):
        """从all_trials重建best_results"""
        if not self.all_trials:
            return None
        
        # 找到最佳trial
        best_trial = max(self.all_trials, key=lambda x: x['value'])
        
        return {
            'best_accuracy': best_trial['value'],
            'best_trial_number': best_trial['number'],
            'best_params': best_trial['params'],
            'completed_trials': len(self.all_trials)
        }
    
    def print_summary(self):
        """打印优化结果摘要"""
        print(f"\n{'='*80}")
        print("Optuna优化结果摘要")
        print(f"{'='*80}")
        
        if self.config:
            print(f"\n实验配置:")
            print(f"  优化器: {self.config.get('optimizer', 'N/A')}")
            print(f"  模型: {self.config.get('model', 'N/A')}")
            print(f"  训练轮数: {self.config.get('epochs', 'N/A')}")
            print(f"  Total Trials: {self.config.get('n_trials', 'N/A')}")
        
        if self.best_results:
            print(f"\n最佳结果:")
            print(f"  准确率: {self.best_results['best_accuracy']:.4f}%")
            print(f"  Trial编号: {self.best_results['best_trial_number']}")
            print(f"  完成的Trials: {self.best_results['completed_trials']}")
            
            print(f"\n最佳参数:")
            for key, value in self.best_results['best_params'].items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.6f}")
                else:
                    print(f"    {key}: {value}")
        
        if self.df_trials is not None:
            print(f"\nTop 5 Trials:")
            top5 = self.df_trials.nlargest(5, 'value')
            for idx, row in top5.iterrows():
                print(f"  Trial {row['number']}: {row['value']:.4f}%")
        
        print(f"{'='*80}\n")
    
    def plot_optimization_history(self, save_path=None):
        """绘制优化历史"""
        if self.df_trials is None:
            print("没有可用的trial数据")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 左图: 所有trials的准确率
        ax1.plot(self.df_trials['number'], self.df_trials['value'], 'o-', alpha=0.6)
        ax1.set_xlabel('Trial Number', fontsize=12)
        ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
        ax1.set_title('Optimization History', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 自适应y轴范围（留10%边距）
        min_acc = self.df_trials['value'].min()
        max_acc = self.df_trials['value'].max()
        margin = (max_acc - min_acc) * 0.1
        ax1.set_ylim(min_acc - margin, max_acc + margin)
        
        # 添加最佳值的水平线
        if self.best_results:
            best_acc = self.best_results['best_accuracy']
            ax1.axhline(y=best_acc, color='r', linestyle='--', 
                       label=f'Best: {best_acc:.4f}%', linewidth=2)
            ax1.legend()
        
        # 右图: 累积最佳准确率
        cumulative_best = self.df_trials['value'].cummax()
        ax2.plot(self.df_trials['number'], cumulative_best, 'g-', linewidth=2)
        ax2.fill_between(self.df_trials['number'], cumulative_best, alpha=0.3)
        ax2.set_xlabel('Trial Number', fontsize=12)
        ax2.set_ylabel('Best Accuracy So Far (%)', fontsize=12)
        ax2.set_title('Cumulative Best Accuracy', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 自适应y轴范围（右图）
        min_cum = cumulative_best.min()
        max_cum = cumulative_best.max()
        margin_cum = (max_cum - min_cum) * 0.1 if max_cum > min_cum else 1.0
        ax2.set_ylim(min_cum - margin_cum, max_cum + margin_cum)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.close()  # 关闭图形，避免内存泄漏
    
    def plot_param_importance(self, save_path=None):
        """绘制参数重要性（通过相关性分析）"""
        if self.all_trials is None or len(self.all_trials) == 0:
            print("没有可用的trial数据")
            return
        
        # 从实际的trial数据中提取参数名
        valid_trials = [t for t in self.all_trials if 'params' in t and t['params']]
        if not valid_trials:
            print("没有找到包含参数的trial数据")
            return
        
        # 获取参数名列表（从第一个valid trial中）
        param_names = list(valid_trials[0]['params'].keys())
        
        if not param_names:
            print("没有找到参数数据")
            return
        
        # 计算每个参数与准确率的相关性
        correlations = {}
        for param in param_names:
            # 从params字典中提取参数值
            param_values = []
            accuracy_values = []
            for trial in valid_trials:
                if param in trial['params']:
                    param_values.append(trial['params'][param])
                    accuracy_values.append(trial['value'])
            
            if len(param_values) < 2:
                continue  # 数据不足，跳过该参数
            
            # 计算相关系数
            corr = np.corrcoef(param_values, accuracy_values)[0, 1]
            correlations[param] = abs(corr)  # 使用绝对值
        
        if not correlations:
            print("没有足够的参数数据来计算重要性")
            return
        
        # 排序
        sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        params, importances = zip(*sorted_params)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
        bars = ax.barh(params, importances, color=colors)
        ax.set_xlabel('Absolute Correlation with Accuracy', fontsize=12)
        ax.set_title('Parameter Importance (Correlation Analysis)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (param, importance) in enumerate(sorted_params):
            ax.text(importance, i, f' {importance:.3f}', 
                   va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.close()  # 关闭图形，避免内存泄漏
    
    def plot_param_distributions(self, save_path=None):
        """绘制参数分布（自适应百分比，确保有足够数据点）"""
        if self.df_trials is None or len(self.all_trials) < 4:
            print("数据不足，无法绘制参数分布（至少需要4个trials）")
            return
        
        # 检查trials是否包含params字段
        valid_trials = [t for t in self.all_trials if 'params' in t and t['params']]
        if not valid_trials:
            print("没有找到包含参数的trial数据")
            return
        
        # 自适应选择百分比，确保每边至少有5个数据点（更清晰的箱线图）
        n_trials = len(valid_trials)
        if n_trials >= 50:
            # 50+个trials: 用10%（每边至少5个）
            percentage = 0.1
            n_samples = max(5, int(n_trials * percentage))
        elif n_trials >= 20:
            # 20-49个trials: 用20%（每边至少4个）
            percentage = 0.2
            n_samples = max(4, int(n_trials * percentage))
        else:
            # <20个trials: 用30%（每边至少2个）
            percentage = 0.3
            n_samples = max(2, int(n_trials * percentage))
        
        sorted_trials = sorted(valid_trials, key=lambda x: x['value'], reverse=True)
        top_trials = sorted_trials[:n_samples]
        bottom_trials = sorted_trials[-n_samples:]
        
        print(f"  使用Top {percentage*100:.0f}% ({n_samples}个) vs Bottom {percentage*100:.0f}% ({n_samples}个)")
        
        # 获取所有参数名
        param_names = list(valid_trials[0]['params'].keys())
        
        # 创建子图
        n_params = len(param_names)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_params > 1 else [axes]
        
        for idx, param in enumerate(param_names):
            ax = axes[idx]
            
            # 提取参数值
            top_values = [t['params'][param] for t in top_trials]
            bottom_values = [t['params'][param] for t in bottom_trials]
            
            # 绘制箱线图
            ax.boxplot([top_values, bottom_values], 
                      labels=[f'Top {percentage*100:.0f}%', f'Bottom {percentage*100:.0f}%'],
                      patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2))
            
            ax.set_title(param, fontsize=12, fontweight='bold')
            ax.set_ylabel('Parameter Value', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Parameter Distributions: Top {percentage*100:.0f}% vs Bottom {percentage*100:.0f}% ({n_samples} trials each)', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.close()  # 关闭图形，避免内存泄漏
    
    def plot_param_relationships(self, param1, param2, save_path=None):
        """绘制两个参数之间的关系"""
        if self.all_trials is None:
            print("没有可用的trial数据")
            return
        
        # 提取参数值和准确率
        param1_values = [t['params'][param1] for t in self.all_trials]
        param2_values = [t['params'][param2] for t in self.all_trials]
        accuracy_values = [t['value'] for t in self.all_trials]
        
        # 绘制散点图
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(param1_values, param2_values, 
                           c=accuracy_values, cmap='viridis',
                           s=100, alpha=0.6, edgecolors='black')
        
        ax.set_xlabel(param1, fontsize=12)
        ax.set_ylabel(param2, fontsize=12)
        ax.set_title(f'{param1} vs {param2}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Validation Accuracy (%)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.close()  # 关闭图形，避免内存泄漏
    
    def generate_report(self):
        """生成完整的可视化报告"""
        print("\n生成可视化报告...")
        
        # 创建保存目录
        viz_dir = self.result_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 生成各种图表
        print("1. 绘制优化历史...")
        self.plot_optimization_history(
            save_path=viz_dir / 'optimization_history.png'
        )
        
        print("2. 绘制参数重要性...")
        self.plot_param_importance(
            save_path=viz_dir / 'param_importance.png'
        )
        
        print("3. 绘制参数分布...")
        self.plot_param_distributions(
            save_path=viz_dir / 'param_distributions.png'
        )
        
        # 保存摘要到文件
        summary_path = viz_dir / 'summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Optuna优化结果摘要\n")
            f.write("="*80 + "\n\n")
            
            if self.config:
                f.write("实验配置:\n")
                for key, value in self.config.items():
                    f.write(f"  {key}: {value}\n")
            
            if self.best_results:
                f.write("\n最佳结果:\n")
                f.write(f"  准确率: {self.best_results['best_accuracy']:.4f}%\n")
                f.write(f"  Trial编号: {self.best_results['best_trial_number']}\n")
                f.write("\n最佳参数:\n")
                for key, value in self.best_results['best_params'].items():
                    if isinstance(value, float):
                        f.write(f"    {key}: {value:.6f}\n")
                    else:
                        f.write(f"    {key}: {value}\n")
        
        print(f"\n报告已生成到: {viz_dir}")
        print(f"  - optimization_history.png")
        print(f"  - param_importance.png")
        print(f"  - param_distributions.png")
        print(f"  - summary.txt")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Optuna结果可视化')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Optuna结果目录路径')
    parser.add_argument('--report', action='store_true',
                       help='生成完整报告')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建可视化器
    visualizer = OptunaVisualizer(args.result_dir)
    
    # 打印摘要
    visualizer.print_summary()
    
    # 如果指定了--report，生成完整报告
    if args.report:
        visualizer.generate_report()
    else:
        # 否则只显示交互式图表
        print("\n提示: 使用 --report 参数可生成完整的可视化报告")
        visualizer.plot_optimization_history()
        visualizer.plot_param_importance()
        visualizer.plot_param_distributions()


if __name__ == '__main__':
    main()
