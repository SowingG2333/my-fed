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
    
    # 注意：这里传递的是原始训练数据集的索引范围
    # 需要将 Subset 对象转换回原始索引空间
    original_train_indices = train_dataset.dataset.indices  # 获取原始索引
    client_datasets, client_dist = dirichlet_partition_mnist(
        dataset=full_dataset,  # 传递完整数据集
        num_clients=num_clients,
        alpha=diri_alpha,
        original_indices=original_train_indices  # 传递原始索引约束
    )
    
    return client_datasets, test_dataset

def dirichlet_partition_mnist(dataset, num_clients, alpha, original_indices=None, seed=42):
    np.random.seed(seed)
    
    # 处理索引范围
    if original_indices is not None:
        all_labels = dataset.targets[original_indices].numpy()
    else:
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
        
        # 改进的样本分配算法
        base_sample = (client_probs[:, class_id] * class_size).astype(int)
        remaining = class_size - base_sample.sum()
        if remaining > 0:
            remaining_idx = np.argsort(-client_probs[:, class_id])
            base_sample[remaining_idx[:remaining]] += 1
        
        # 分配样本
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + base_sample[client_id]
            client_datasets[client_id].extend(
                class_indices[class_id][start_idx:end_idx]
            )
            client_dist[client_id, class_id] = base_sample[client_id]
            start_idx = end_idx
    
    # 转换为Subset对象（保持与原始数据集的索引关联）
    client_subsets = [
        Subset(dataset, indices) 
        for indices in client_datasets
    ]
    
    # 标准化分布
    client_dist = client_dist / client_dist.sum(axis=1, keepdims=True)
    
    return client_subsets, client_dist

# def non_iid_split(dataset, num_clients, classes_per_client=10):
#     """支持原始Dataset和Subset的统一处理"""
#     # 获取实际数据引用和索引
#     if isinstance(dataset, Subset):
#         main_dataset = dataset.dataset
#         indices = dataset.indices
#     else:
#         main_dataset = dataset
#         indices = np.arange(len(dataset))
    
#     # 获取标签数据（兼容不同数据集结构）
#     if hasattr(main_dataset, 'targets'):
#         labels = np.array(main_dataset.targets)[indices]
#     elif hasattr(main_dataset, 'labels'):
#         labels = np.array(main_dataset.labels)[indices]
#     else:
#         raise AttributeError("Dataset missing both 'targets' and 'labels' attributes")
    
#     # 非独立同分布划分逻辑
#     class_indices = [np.where(labels == i)[0] for i in range(10)]
#     client_indices = []
    
#     for _ in range(num_clients):
#         selected_classes = np.random.choice(10, classes_per_client, replace=False)
#         indices = []
#         for cls in selected_classes:
#             take_num = np.random.randint(100, 500)
#             indices.extend(np.random.choice(class_indices[cls], take_num, replace=False))
#         client_indices.append(indices)
    
#     return [Subset(main_dataset, indices) for indices in client_indices]