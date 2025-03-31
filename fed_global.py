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

def data_generate(num_clients, diri_alpha):
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
    
    # 客户端数据划分（基于训练子集）
    client_datasets, client_dist = dirichlet_partition_mnist(
        dataset=train_dataset,  # 直接传入训练子集
        num_clients=num_clients,
        alpha=diri_alpha
    )
    
    return client_datasets, test_dataset

def dirichlet_partition_mnist(dataset, num_clients, alpha, seed=42):
    np.random.seed(seed)
    
    # 处理数据集类型（支持原始数据集或Subset）
    if isinstance(dataset, Subset):
        original_indices = dataset.indices
        all_labels = dataset.dataset.targets[original_indices].numpy()
    else:
        original_indices = None
        all_labels = dataset.targets.numpy()
    
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
    
    # 转换为Subset对象（保持与原始数据集的索引关联）
    if isinstance(dataset, Subset):
        client_subsets = [Subset(dataset.dataset, indices) for indices in client_datasets]
    else:
        client_subsets = [Subset(dataset, indices) for indices in client_datasets]
    
    client_dist = client_dist / client_dist.sum(axis=1, keepdims=True)
    return client_subsets, client_dist