import torchvision                                                  # 引入torchvision库
import torchvision.transforms as transforms                         # 引入torchvision的数据预处理模块
import numpy as np                                                  # 引入numpy库
import torch.nn as nn                                               # 引入torch的神经网络模块
from torch.utils.data import Subset                                 # 引入torch的子数据集模块
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST       # 引入torchvision的MNIST和CIFAR10数据集
import torch                                                        # 引入torch库
from torch.utils.data import random_split                           # 引入torch的随机划分数据集模块
import torch.nn.functional as F

class MNIST_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 短路连接（输入输出维度不一致时使用1x1卷积调整）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        return F.relu(out)

class Fashion_Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # 28x28 -> 14x14
            ResBlock(32, 64),                                     # 14x14 -> 14x14
            nn.MaxPool2d(2),                                      # 14x14 -> 7x7
            ResBlock(64, 128, stride=2),                          # 7x7 -> 4x4（进一步提取特征）
            nn.AdaptiveAvgPool2d((1, 1))                          # 4x4 -> 1x1（替代全连接层）
        )
        
        # 分类器（轻量设计）
        self.classifier = nn.Linear(128, num_classes)

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Kaiming初始化（适配ReLU）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class CIFAR10_AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 阶段1: 32x32 → 16x16
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 保持尺寸
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            
            # 阶段2: 16x16 → 8x8
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            
            # 阶段3: 8x8 → 8x8 (无池化)
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            
            # 阶段4: 8x8 → 4x4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
        )
        
        # 轻量级分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def data_generate(num_clients, train_per, diri_alpha, data_type='MNIST', device=None):
    # 加载原始数据集
    if data_type == 'MNIST':
        full_dataset = MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
    elif data_type == 'FashionMNIST':
        full_dataset = torchvision.datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        )
    elif data_type == 'CIFAR10':
        full_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)), 
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        )
    
    # 划分训练验证集
    train_size = int(train_per * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # 客户端数据划分
    client_datasets = dirichlet_partition(
        dataset=train_dataset,
        num_clients=num_clients,
        alpha=diri_alpha,
        data_type=data_type
    )
    
    return client_datasets, test_dataset

def dirichlet_partition(dataset, num_clients, alpha, data_type, seed=42):
    np.random.seed(seed)
    
    # 处理数据集类型（支持原始数据集或Subset）
    if data_type == 'MNIST' or 'FashionMNIST':
        if isinstance(dataset, Subset):
            original_indices = dataset.indices
            all_labels = dataset.dataset.targets[original_indices].numpy()
        else:
            original_indices = None
            all_labels = dataset.targets.numpy()
    elif data_type == 'CIFAR10':
        if isinstance(dataset, Subset):
            original_indices = dataset.indices
            all_labels = np.array(dataset.dataset.targets)[original_indices].flatten()
        else:
            all_labels = np.array(dataset.targets).flatten()
    
    num_classes = 10
    class_indices = [np.where(all_labels == i)[0] for i in range(num_classes)]
    
    # 生成Dirichlet分布
    client_probs = np.random.dirichlet([alpha]*num_classes, size=num_clients)
    
    client_datasets = [[] for _ in range(num_clients)]
    client_dist = np.zeros((num_clients, num_classes))
    
    for class_id in range(num_classes):
        np.random.shuffle(class_indices[class_id])
        class_size = len(class_indices[class_id])
        
        base_sample = (client_probs[:, class_id] * class_size).astype(int)
        remaining = class_size - base_sample.sum()
        if remaining > 0:
            remaining_idx = np.argsort(-client_probs[:, class_id])
            base_sample[remaining_idx[:remaining]] += 1
        
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + base_sample[client_id]
            client_datasets[client_id].extend(class_indices[class_id][start_idx:end_idx])
            client_dist[client_id, class_id] = base_sample[client_id]
            start_idx = end_idx
    
    # 转换为Subset对象
    if isinstance(dataset, Subset):
        client_subsets = [Subset(dataset.dataset, indices) for indices in client_datasets]
    else:
        client_subsets = [Subset(dataset, indices) for indices in client_datasets]
    
    return client_subsets