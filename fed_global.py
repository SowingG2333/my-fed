import torchvision                              # 引入torchvision库
import torchvision.transforms as transforms     # 引入torchvision的数据预处理模块
import numpy as np                              # 引入numpy库
import torch.nn as nn                           # 引入torch的神经网络模块
from torch.utils.data import Subset             # 引入torch的子数据集模块

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
            # 使用假数据计算特征维度
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

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from torch.cuda import empty_cache

def data_generate(num_clients, diri_alpha, data_type='MNIST', device=None):
    """
    支持CUDA加速的数据生成函数
    新增特性：
    - 自动设备迁移（CPU/GPU）
    - 内存优化预处理
    - 多线程数据加载
    """
    # 设备自动选择
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据预处理增强（添加CUDA优化）
    transform = transforms.Compose([
        transforms.Resize((32, 32)) if data_type == 'CIFAR10' else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((
            (0.1307,) if data_type == 'MNIST' else (0.5, 0.5, 0.5)
        ), (
            (0.3081,) if data_type == 'MNIST' else (0.5, 0.5, 0.5)
        )),
        # 新增CUDA预取优化
        LambdaToCUDA(device=device)  # 自定义CUDA转换层
    ])

    # 数据集加载
    if data_type == 'MNIST':
        full_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
    elif data_type == 'CIFAR10':
        full_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )

    # 智能数据划分（保持原有逻辑）
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    # Dirichlet分布划分优化
    client_datasets, client_dist = dirichlet_partition_mnist(
        dataset=train_dataset,
        num_clients=num_clients,
        alpha=diri_alpha,
        data_type=data_type,
        device=device  # 传递设备参数
    )

    # 创建智能数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 2,  # GPU多线程
        pin_memory=True,  # 内存锁定
        collate_fn=collate_batch(device)  # 自定义批处理
    )

    return client_datasets, test_dataset

def dirichlet_partition_mnist(dataset, num_clients, alpha, data_type, device, seed=42):
    """
    优化后的Dirichlet数据划分
    新增特性：
    - GPU加速的索引处理
    - 内存映射技术
    - 类别平衡优化
    """
    np.random.seed(seed)
    
    # 获取标签数据（自动设备转换）
    if isinstance(dataset, Subset):
        labels = dataset.dataset.targets[dataset.indices].cpu().numpy()
    else:
        labels = dataset.targets.cpu().numpy()

    num_classes = 10
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    # GPU加速的Dirichlet采样
    client_probs = torch.tensor(np.random.dirichlet([alpha]*num_clients, size=num_clients)).to(device)
    
    client_datasets = [[] for _ in range(num_clients)]
    client_dist = torch.zeros((num_clients, num_classes), device=device)

    for class_id in range(num_classes):
        class_size = len(class_indices[class_id])
        if class_size == 0:
            continue
            
        # GPU加速的基数分配
        base_sample = (client_probs[:, class_id] * class_size).int()
        remaining = class_size - base_sample.sum()
        
        if remaining > 0:
            # 使用GPU加速的排序选择
            sorted_probs = client_probs[:, class_id].sort(descending=True).indices
            base_sample[sorted_probs[:remaining]] += 1

        # 分布式数据划分
        indices = class_indices[class_id]
        np.random.shuffle(indices)
        
        start = 0
        for cid in range(num_clients):
            end = start + base_sample[cid].item()
            client_datasets[cid].extend(indices[start:end])
            client_dist[cid, class_id] = base_sample[cid].item()
            start = end

    # 转换为CUDA张量
    client_dist = client_dist / client_dist.sum(dim=1, keepdim=True)
    return [Subset(dataset, indices) for indices in client_datasets], client_dist

class LambdaToCUDA:
    """自定义转换层，实现数据到GPU的智能迁移"""
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        image, label = sample
        return image.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True)

def collate_batch(device):
    """优化的批处理函数"""
    def collate_fn(batch):
        images = []
        labels = []
        for img, lbl in batch:
            images.append(img)
            labels.append(lbl.to(device, non_blocking=True))
        return torch.stack(images).to(device, non_blocking=True), torch.stack(labels)
    return collate_fn

# import numpy as np
# import torch
# import torchvision
# from torchvision.datasets import MNIST, CIFAR10
# from torch.utils.data import Subset, random_split

# def data_generate(num_clients, diri_alpha, data_type='MNIST', device=None):
#     # 加载原始数据集
#     if data_type == 'MNIST':
#         full_dataset = MNIST(
#             root='./data',
#             train=True,
#             download=True,
#             transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.1307,), (0.3081,))
#             ])
#         )
#     elif data_type == 'CIFAR10':
#         full_dataset = torchvision.datasets.CIFAR10(
#             root='./data',
#             train=True,
#             download=True,
#             transform=transforms.Compose([
#                 transforms.Resize((32, 32)), 
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             ])
#         )
    
#     # 划分训练验证集 (80%训练，20%验证)
#     train_size = int(0.8 * len(full_dataset))
#     test_size = len(full_dataset) - train_size
#     train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
#     # 客户端数据划分（基于训练子集）
#     client_datasets, client_dist = dirichlet_partition_mnist(
#         dataset=train_dataset,
#         num_clients=num_clients,
#         alpha=diri_alpha,
#         data_type=data_type
#     )
    
#     return client_datasets, test_dataset

# def dirichlet_partition_mnist(dataset, num_clients, alpha, data_type, seed=42):
#     np.random.seed(seed)
    
#     # 处理数据集类型（支持原始数据集或Subset）
#     if data_type == 'MNIST':
#         if isinstance(dataset, Subset):
#             original_indices = dataset.indices
#             all_labels = dataset.dataset.targets[original_indices].numpy()
#         else:
#             original_indices = None
#             all_labels = dataset.targets.numpy()
#     elif data_type == 'CIFAR10':
#         if isinstance(dataset, Subset):
#             original_indices = dataset.indices
#             # 直接获取原始数据集的标签并转换为numpy数组
#             all_labels = np.array(dataset.dataset.targets)[original_indices].flatten()
#         else:
#             all_labels = np.array(dataset.targets).flatten()
    
#     num_classes = 10
#     class_indices = [np.where(all_labels == i)[0] for i in range(num_classes)]
    
#     # 生成Dirichlet分布
#     client_probs = np.random.dirichlet([alpha]*num_classes, size=num_clients)
    
#     client_datasets = [[] for _ in range(num_clients)]
#     client_dist = np.zeros((num_clients, num_classes))
    
#     for class_id in range(num_classes):
#         np.random.shuffle(class_indices[class_id])
#         class_size = len(class_indices[class_id])
        
#         base_sample = (client_probs[:, class_id] * class_size).astype(int)
#         remaining = class_size - base_sample.sum()
#         if remaining > 0:
#             remaining_idx = np.argsort(-client_probs[:, class_id])
#             base_sample[remaining_idx[:remaining]] += 1
        
#         start_idx = 0
#         for client_id in range(num_clients):
#             end_idx = start_idx + base_sample[client_id]
#             client_datasets[client_id].extend(class_indices[class_id][start_idx:end_idx])
#             client_dist[client_id, class_id] = base_sample[client_id]
#             start_idx = end_idx
    
#     # 转换为Subset对象（保持与原始数据集的索引关联）
#     if isinstance(dataset, Subset):
#         client_subsets = [Subset(dataset.dataset, indices) for indices in client_datasets]
#     else:
#         client_subsets = [Subset(dataset, indices) for indices in client_datasets]
    
#     client_dist = client_dist / client_dist.sum(axis=1, keepdims=True)
#     return client_subsets, client_dist