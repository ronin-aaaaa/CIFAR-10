"""
数据增强工具
包含Cutout、Mixup、RandomErasing等高级数据增强技术
"""
import torch
import numpy as np
from PIL import Image


class Cutout:
    """
    Cutout数据增强
    随机遮挡图像的一部分，提高模型鲁棒性
    """
    
    def __init__(self, n_holes=1, length=16):
        """
        参数:
            n_holes: 遮挡区域数量
            length: 遮挡区域边长
        """
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): 形状为 (C, H, W) 的张量图像
        Returns:
            Tensor: 应用Cutout后的图像
        """
        h = img.size(1)
        w = img.size(2)

        # 使用 torch 代替 numpy，避免兼容性问题
        mask = torch.ones((h, w), dtype=torch.float32)

        for n in range(self.n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()

            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)

            mask[y1: y2, x1: x2] = 0.

        mask = mask.expand_as(img)
        img = img * mask

        return img


class RandomErasing:
    """
    Random Erasing数据增强
    以一定概率随机擦除图像区域
    """
    
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        """
        参数:
            probability: 应用擦除的概率
            sl: 最小擦除面积比例
            sh: 最大擦除面积比例
            r1: 最小宽高比
            mean: 填充的均值
        """
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        # 使用 torch 代替 numpy
        if torch.rand(1).item() >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = (torch.rand(1).item() * (self.sh - self.sl) + self.sl) * area
            aspect_ratio = torch.rand(1).item() * (1 / self.r1 - self.r1) + self.r1

            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = torch.randint(0, img.size()[1] - h, (1,)).item()
                y1 = torch.randint(0, img.size()[2] - w, (1,)).item()
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def mixup_data(x, y, alpha=1.0, device='cuda'):
    """
    Mixup数据增强
    混合两个样本及其标签
    
    参数:
        x: 输入数据
        y: 标签
        alpha: Beta分布参数
        device: 设备
    
    返回:
        mixed_x: 混合后的数据
        y_a: 第一个标签
        y_b: 第二个标签
        lam: 混合比例
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup损失函数
    
    参数:
        criterion: 基础损失函数
        pred: 模型预测
        y_a: 第一个标签
        y_b: 第二个标签
        lam: 混合比例
    
    返回:
        混合损失
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CutMix:
    """
    CutMix数据增强
    裁剪并粘贴图像块
    """
    
    def __init__(self, beta=1.0):
        self.beta = beta

    def __call__(self, x, y):
        """
        参数:
            x: 输入数据 (batch_size, C, H, W)
            y: 标签 (batch_size,)
        
        返回:
            mixed_x: 混合后的数据
            y_a: 第一个标签
            y_b: 第二个标签
            lam: 混合比例
        """
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(x.size()[0]).cuda()
        
        y_a = y
        y_b = y[rand_index]
        
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        # 调整lambda以匹配像素比例
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        
        return x, y_a, y_b, lam

    def rand_bbox(self, size, lam):
        """生成随机边界框"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # 随机中心点
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


class AutoAugment:
    """
    AutoAugment for CIFAR-10
    使用预定义的增强策略
    """
    
    def __init__(self):
        from torchvision import transforms
        self.policies = [
            # 策略1
            [(transforms.ColorJitter(brightness=0.4), 0.9, None),
             (transforms.RandomRotation(15), 0.8, None)],
            # 策略2
            [(transforms.ColorJitter(contrast=0.4), 0.9, None),
             (transforms.RandomAffine(0, translate=(0.1, 0.1)), 0.8, None)],
        ]

    def __call__(self, img):
        """随机应用一个策略"""
        policy = np.random.choice(self.policies)
        for transform, prob, magnitude in policy:
            if np.random.random() < prob:
                img = transform(img)
        return img
