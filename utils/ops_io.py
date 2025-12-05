"""
数据加载和IO工具
包含CIFAR-10数据集的加载、预处理和数据增强
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import os
from utils.ops_augment import Cutout, RandomErasing


class CIFAR10DataLoader:
    """CIFAR-10数据加载器"""
    
    def __init__(self, data_dir='./data', batch_size=128, num_workers=4, 
                 use_cutout=True, use_random_erasing=False, validation_split=0.1):
        """
        参数:
            data_dir: 数据存储目录
            batch_size: 批次大小
            num_workers: 数据加载线程数
            use_cutout: 是否使用Cutout增强
            use_random_erasing: 是否使用Random Erasing增强
            validation_split: 验证集比例
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        
        # CIFAR-10的均值和标准差
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)
        
        # 构建数据增强
        self.train_transform = self._build_train_transform(use_cutout, use_random_erasing)
        self.test_transform = self._build_test_transform()
        
    def _build_train_transform(self, use_cutout, use_random_erasing):
        """构建训练数据增强"""
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ]
        
        if use_cutout:
            transform_list.append(Cutout(n_holes=1, length=16))
        
        if use_random_erasing:
            transform_list.append(RandomErasing(probability=0.5, mean=self.mean))
        
        return transforms.Compose(transform_list)
    
    def _build_test_transform(self):
        """构建测试数据增强"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def get_train_valid_loader(self, shuffle=True, random_seed=42):
        """
        获取训练集和验证集的DataLoader
        
        返回:
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
        """
        # 加载训练数据
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # 创建验证集
        valid_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.test_transform
        )
        
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.validation_split * num_train))
        
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        # 添加persistent_workers避免多进程序列化问题
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        return train_loader, valid_loader
    
    def get_test_loader(self):
        """
        获取测试集的DataLoader
        
        返回:
            test_loader: 测试数据加载器
        """
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        return test_loader
    
    def get_classes(self):
        """获取CIFAR-10的类别名称"""
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']


def save_checkpoint(state, filename='checkpoint.pth'):
    """
    保存模型检查点
    
    参数:
        state: 包含模型状态、优化器状态等信息的字典
        filename: 保存文件名
    """
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename='checkpoint.pth', device='cuda'):
    """
    加载模型检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        filename: 检查点文件名
        device: 设备
    
    返回:
        start_epoch: 开始的epoch
        best_acc: 最佳准确率
    """
    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename, map_location=device)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {start_epoch} with best accuracy {best_acc:.2f}%")
        return start_epoch, best_acc
    else:
        print(f"No checkpoint found at {filename}")
        return 0, 0.0


def save_model(model, filename='best_model.pth'):
    """
    保存最佳模型
    
    参数:
        model: 模型
        filename: 保存文件名
    """
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_model(model, filename='best_model.pth', device='cuda'):
    """
    加载模型权重
    
    参数:
        model: 模型
        filename: 模型文件名
        device: 设备
    """
    if os.path.isfile(filename):
        print(f"Loading model from {filename}")
        model.load_state_dict(torch.load(filename, map_location=device))
        print("Model loaded successfully")
    else:
        print(f"No model found at {filename}")
