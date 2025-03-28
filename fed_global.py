import torch                            # 引入torch库
import numpy as np                      # 引入numpy库
import torch.nn as nn                   # 引入torch的神经网络模块
from torch.utils.data import Subset     # 引入torch的子数据集模块

class FedModel(nn.Module):
    """联邦学习全局模型"""
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
    
import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import Subset, random_split

def data_split(num_clients):
    # 加载原始MNIST数据集
    full_dataset = MNIST(
        root='./data',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    
    # 划分训练验证集 (80%训练，20%验证)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # 对训练集进行非独立同分布划分
    client_datasets = non_iid_split(train_dataset, num_clients)
    
    return client_datasets, test_dataset

def non_iid_split(dataset, num_clients, classes_per_client=10):
    """支持原始Dataset和Subset的统一处理"""
    # 获取实际数据引用和索引
    if isinstance(dataset, Subset):
        main_dataset = dataset.dataset
        indices = dataset.indices
    else:
        main_dataset = dataset
        indices = np.arange(len(dataset))
    
    # 获取标签数据（兼容不同数据集结构）
    if hasattr(main_dataset, 'targets'):
        labels = np.array(main_dataset.targets)[indices]
    elif hasattr(main_dataset, 'labels'):
        labels = np.array(main_dataset.labels)[indices]
    else:
        raise AttributeError("Dataset missing both 'targets' and 'labels' attributes")
    
    # 非独立同分布划分逻辑
    class_indices = [np.where(labels == i)[0] for i in range(10)]
    client_indices = []
    
    for _ in range(num_clients):
        selected_classes = np.random.choice(10, classes_per_client, replace=False)
        indices = []
        for cls in selected_classes:
            take_num = np.random.randint(100, 500)
            indices.extend(np.random.choice(class_indices[cls], take_num, replace=False))
        client_indices.append(indices)
    
    return [Subset(main_dataset, indices) for indices in client_indices]