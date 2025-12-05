"""
训练和测试工具
包含训练、验证、测试等核心方法
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import time
import numpy as np
from tqdm import tqdm
from utils.ops_augment import mixup_data, mixup_criterion


class Trainer:
    """模型训练器"""
    
    def __init__(self, model, train_loader, valid_loader, device='cuda',
                 optimizer_name='sgd', learning_rate=0.1, weight_decay=5e-4, epochs=200,
                 use_mixup=True, mixup_alpha=1.0, label_smoothing=0.1,
                 scheduler_type='cosine', log_file=None):
        """
        参数:
            model: 神经网络模型
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            device: 训练设备
            optimizer_name: 优化器名称 ('sgd', 'adam', 'adamw', 'rmsprop')
            learning_rate: 初始学习率
            weight_decay: 权重衰减
            epochs: 训练轮数
            use_mixup: 是否使用Mixup
            mixup_alpha: Mixup的alpha参数
            label_smoothing: 标签平滑参数
            scheduler_type: 学习率调度器类型 ('cosine' or 'multistep')
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.epochs = epochs
        self.use_mixup = use_mixup
        self.optimizer_name = optimizer_name.lower()
        
        # 为特定优化器调整Mixup强度
        if self.optimizer_name == 'adamw' and use_mixup:
            self.mixup_alpha = max(mixup_alpha, 1.0)  # AdamW使用最强Mixup=1.0对抗严重过拟合
        else:
            self.mixup_alpha = mixup_alpha
        
        # 损失函数（使用标签平滑）
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # 创建优化器
        self.optimizer = self._create_optimizer(learning_rate, weight_decay)
        
        # 学习率调度器
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        elif scheduler_type == 'multistep':
            self.scheduler = MultiStepLR(self.optimizer, milestones=[60, 120, 160], gamma=0.2)
        else:
            self.scheduler = None
        
        # 训练历史
        self.train_losses = []
        self.train_accs = []
        self.valid_losses = []
        self.valid_accs = []
        self.learning_rates = []
        self.best_acc = 0.0
        
        # 日志文件
        self.log_file = log_file
    
    def _log(self, message):
        """输出日志到控制台和文件"""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
    
    def _create_optimizer(self, learning_rate, weight_decay):
        """创建优化器"""
        if self.optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True
                # nesterov=True # Nesterov动量，加速收敛，提高准确率，但会增加训练时间
            )
        elif self.optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.optimizer_name == 'rmsprop':
            return optim.RMSprop(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                alpha=0.99,      # 经典配置
                eps=1e-8,
                momentum=0,      # RMSprop原生不需要momentum(积累之前的梯度信息来加速训练)
                centered=False
            )
        else:
            raise ValueError(f"不支持的优化器: {self.optimizer_name}")
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        print()  # 添加换行让输出更清晰
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs} [Train]')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Mixup数据增强
            if self.use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, self.mixup_alpha, self.device
                )
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            else:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            
            if self.use_mixup:
                correct += (lam * predicted.eq(targets_a).sum().item() + 
                           (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.3f}',
                'Acc': f'{acc:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        epoch_loss = train_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        valid_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.valid_loader, desc=f'Epoch {epoch}/{self.epochs} [Valid]')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 更新进度条
                acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{valid_loss/(batch_idx+1):.3f}',
                    'Acc': f'{acc:.2f}%'
                })
        
        epoch_loss = valid_loss / len(self.valid_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, save_path='checkpoints'):
        """完整训练流程"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Starting training for {self.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixup: {self.use_mixup}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 验证
            valid_loss, valid_acc = self.validate(epoch)
            self.valid_losses.append(valid_loss)
            self.valid_accs.append(valid_acc)
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 打印统计信息
            self._log(f"\nEpoch {epoch}/{self.epochs}:")
            self._log(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            self._log(f"  Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%")
            
            # 保存最佳模型
            if valid_acc > self.best_acc:
                self._log(f"  ✓ New best accuracy: {valid_acc:.2f}% (previous: {self.best_acc:.2f}%)")
                self.best_acc = valid_acc
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': self.best_acc,
                    'train_losses': self.train_losses,
                    'train_accs': self.train_accs,
                    'valid_losses': self.valid_losses,
                    'valid_accs': self.valid_accs,
                    'learning_rates': self.learning_rates,
                }
                torch.save(checkpoint, f'{save_path}/best_model.pth')
            
            # 定期保存检查点
            if epoch % 50 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': self.best_acc,
                }
                torch.save(checkpoint, f'{save_path}/checkpoint_epoch_{epoch}.pth')
        
        training_time = time.time() - start_time
        self._log(f"\n{'='*60}")
        self._log(f"Training completed in {training_time/3600:.2f} hours")
        self._log(f"Best validation accuracy: {self.best_acc:.2f}%")
        self._log(f"{'='*60}\n")
        
        # 保存最终总结
        if save_path:
            self._save_final_summary(save_path, training_time)
        
        return self.best_acc
    
    def _save_final_summary(self, save_path, training_time):
        """保存最终训练总结"""
        summary_file = f'{save_path}/training_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("训练总结\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"优化器: {self.optimizer_name.upper()}\n")
            f.write(f"总训练轮数: {len(self.train_losses)}\n")
            f.write(f"总用时: {training_time/3600:.2f} 小时\n")
            f.write(f"平均每轮时间: {training_time/len(self.train_losses):.1f} 秒\n\n")
            
            f.write("最终结果:\n")
            f.write(f"  最佳验证准确率: {self.best_acc:.2f}%\n")
            f.write(f"  最终训练准确率: {self.train_accs[-1]:.2f}%\n")
            f.write(f"  最终验证准确率: {self.valid_accs[-1]:.2f}%\n\n")
            
            f.write("注: 详细训练过程请查看 training_log.txt\n")
            f.write("    完整训练历史已保存在 best_model.pth 中\n")


class Tester:
    """模型测试器"""
    
    def __init__(self, model, test_loader, device='cuda'):
        """
        参数:
            model: 神经网络模型
            test_loader: 测试数据加载器
            device: 测试设备
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def test(self, save_path=None):
        """测试模型"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        # 用于计算每个类别的准确率
        class_correct = [0] * 10
        class_total = [0] * 10
        
        print("\nTesting model...")
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 计算每个类别的准确率
                c = predicted.eq(targets)
                for i in range(len(targets)):
                    label = targets[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                
                # 更新进度条
                pbar.set_postfix({'Acc': f'{100.*correct/total:.2f}%'})
        
        # 打印结果
        test_acc = 100. * correct / total
        test_loss_avg = test_loss / len(self.test_loader)
        print(f"\n{'='*60}")
        print(f"Test Results:")
        print(f"  Overall Accuracy: {test_acc:.2f}%")
        print(f"  Average Loss: {test_loss_avg:.4f}")
        
        # 打印每个类别的准确率
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
        print(f"\nPer-class Accuracy:")
        for i in range(10):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f"  {classes[i]:12s}: {acc:.2f}%")
        print(f"{'='*60}\n")
        
        # 如果指定了保存路径，将测试结果追加到training_summary.txt
        if save_path:
            summary_file = f'{save_path}/training_summary.txt'
            try:
                with open(summary_file, 'a', encoding='utf-8') as f:
                    f.write("\n" + "="*60 + "\n")
                    f.write("测试集结果\n")
                    f.write("="*60 + "\n\n")
                    f.write(f"测试集准确率: {test_acc:.2f}%\n")
                    f.write(f"测试集平均损失: {test_loss_avg:.4f}\n\n")
                    f.write("各类别准确率:\n")
                    for i in range(10):
                        if class_total[i] > 0:
                            acc = 100 * class_correct[i] / class_total[i]
                            f.write(f"  {classes[i]:12s}: {acc:5.2f}%\n")
                print(f"测试结果已追加到: {summary_file}")
            except Exception as e:
                print(f"保存测试结果失败: {e}")
        
        return test_acc


def evaluate_model(model, data_loader, device='cuda'):
    """
    快速评估模型准确率
    
    参数:
        model: 模型
        data_loader: 数据加载器
        device: 设备
    
    返回:
        accuracy: 准确率
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy
