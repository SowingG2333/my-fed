import random                           # 引入随机模块
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
    
# 非独立同分布数据划分
def non_iid_split(dataset, num_clients, classes_per_client=10):
    labels = np.array(dataset.targets)
    class_indices = [np.where(labels == i)[0] for i in range(10)]
    
    client_indices = []
    for _ in range(num_clients):
        selected_classes = np.random.choice(10, classes_per_client, replace=False)
        indices = []
        for cls in selected_classes:
            take_num = np.random.randint(100, 500)
            indices.extend(np.random.choice(class_indices[cls], take_num, replace=False))
        client_indices.append(indices)
    
    return [Subset(dataset, indices) for indices in client_indices]

# 均匀数据划分
def split_dataset_randomly(dataset, num_clients):
    """
    将数据集随机分配给多个客户端，并返回每个客户端的数据子集
    :param dataset: 原始数据集
    :param num_clients: 客户端数量
    :return: 每个客户端的数据子集列表
    """
    data_size = len(dataset)
    indices = list(range(data_size))
    random.shuffle(indices)  # 随机打乱索引

    # 计算每个客户端应分配的平均数据量
    avg_size = data_size // num_clients
    client_data_sizes = [avg_size] * num_clients

    # 在平均数据量的基础上，添加一定的随机性
    for i in range(data_size % num_clients):
        client_data_sizes[i] += 1                               # 分配剩余的数据

    # 确保每个客户端的数据量在一定范围内波动
    for i in range(num_clients):
        if i < num_clients - 1:
            fluctuation = random.randint(-avg_size // 2, avg_size // 2)
            client_data_sizes[i] += fluctuation
            client_data_sizes[-1] -= fluctuation

    client_datasets = []
    start_idx = 0
    for size in client_data_sizes:
        end_idx = start_idx + size
        client_indices = indices[start_idx:end_idx]
        client_subset = [dataset[i] for i in client_indices]    # 直接获取数据子集
        client_datasets.append(client_subset)
        start_idx = end_idx

    return client_datasets                                      # 返回每个客户端的数据子集列表