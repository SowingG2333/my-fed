import torchvision                                  # 引入torchvision库
import torchvision.transforms as transforms         # 引入torchvision的数据预处理模块
import numpy as np                                  # 引入numpy库
import torch.nn as nn                               # 引入torch的神经网络模块
from torch.utils.data import Subset                 # 引入torch的子数据集模块
from torchvision.datasets import MNIST, CIFAR10     # 引入torchvision的MNIST和CIFAR10数据集
import torch                                        # 引入torch库
from torch.utils.data import random_split           # 引入torch的随机划分数据集模块

# class FedModel(nn.Module):
#     """联邦学习全局模型"""
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.layers = nn.Sequential(
#             nn.Linear(784, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 10)
#         )
    
#     def forward(self, x):
#         x = self.flatten(x)
#         return self.layers(x)
    
class FedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 动态计算全连接层输入维度
        with torch.no_grad():
            sample = torch.zeros(1, 3, 32, 32)
            features = self.features(sample)
            self.fc_input_dim = features.view(1, -1).size(1)
        
        # 全连接层部分
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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
    elif data_type == 'CIFAR10':
        full_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
    if data_type == 'MNIST':
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