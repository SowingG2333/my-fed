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
def non_iid_split(dataset, num_clients, classes_per_client=2):
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