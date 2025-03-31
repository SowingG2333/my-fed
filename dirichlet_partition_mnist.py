import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

def dirichlet_partition_mnist(dataset, num_clients, alpha, seed=42):
    """
    使用狄利克雷分布对MNIST数据集进行非IID划分
    
    参数:
        dataset: MNIST数据集实例
        num_clients: 客户端数量
        alpha: 狄利克雷分布的α参数，控制数据分布的异质性程度
               - 小α值(如0.1)会产生高度不平衡的分布
               - 大α值(如100)会使分布接近均匀
        seed: 随机种子
        
    返回:
        client_datasets: 包含每个客户端数据集的列表
        client_dist: 每个客户端的标签分布
    """
    np.random.seed(seed)
    
    # 获取所有样本的标签
    if isinstance(dataset, MNIST):
        all_labels = dataset.targets.numpy()
    else:
        all_labels = np.array(dataset.targets)
    
    # 计算每个类别的索引
    num_classes = 10  # MNIST有10个类别
    class_indices = [np.where(all_labels == i)[0] for i in range(num_classes)]
    
    # 为每个客户端采样狄利克雷分布
    client_distributions = np.random.dirichlet(alpha=[alpha] * num_classes, size=num_clients)
    
    # 初始化客户端数据集
    client_datasets = [[] for _ in range(num_clients)]
    client_dist = np.zeros((num_clients, num_classes))
    
    # 分配每个类别的样本
    for class_id in range(num_classes):
        # 打乱该类别的样本索引
        np.random.shuffle(class_indices[class_id])
        
        # 计算每个客户端应获得的样本数量
        class_size = len(class_indices[class_id])
        client_sample_sizes = np.floor(client_distributions[:, class_id] * class_size).astype(int)
        
        # 处理因舍入导致的样本数不足
        shortage = class_size - np.sum(client_sample_sizes)
        # 将剩余样本分配给概率最高的客户端
        if shortage > 0:
            sorted_clients = np.argsort(-client_distributions[:, class_id])
            for i in range(shortage):
                client_sample_sizes[sorted_clients[i]] += 1
        
        # 分配样本
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + client_sample_sizes[client_id]
            client_indices = class_indices[class_id][start_idx:end_idx]
            client_datasets[client_id].extend(client_indices)
            
            # 记录实际分配的样本数量
            client_dist[client_id, class_id] = len(client_indices)
            start_idx = end_idx
    
    # 将索引列表转换为Subset对象
    client_subsets = [Subset(dataset, indices) for indices in client_datasets]
    
    # 将每个客户端的标签分布转换为概率分布
    for i in range(num_clients):
        if np.sum(client_dist[i]) > 0:  # 避免除以0
            client_dist[i] = client_dist[i] / np.sum(client_dist[i])
    
    return client_subsets, client_dist

def visualize_distribution(client_dist, title="客户端数据分布"):
    """
    可视化客户端数据分布
    
    参数:
        client_dist: 客户端标签分布矩阵，shape为(num_clients, num_classes)
        title: 图表标题
    """
    num_clients, num_classes = client_dist.shape
    
    plt.figure(figsize=(12, 8))
    for i in range(num_clients):
        plt.subplot(int(np.ceil(num_clients/3)), 3, i+1)
        plt.bar(range(num_classes), client_dist[i])
        plt.title(f"客户端 {i+1}")
        plt.xlabel("类别")
        plt.ylabel("比例")
        plt.xticks(range(num_classes))
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()

# 使用示例
def example_usage():
    """
    示例：如何使用上述函数
    """
    from torchvision import datasets, transforms
    
    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # 划分数据集
    num_clients = 5
    alpha = 0.5  # 调整此值可改变分布的不平衡程度
    
    client_datasets, client_dist = dirichlet_partition_mnist(mnist_train, num_clients, alpha)
    
    # 打印每个客户端的数据集大小
    for i, dataset in enumerate(client_datasets):
        print(f"客户端 {i+1} 的数据集大小: {len(dataset)}")
    
    # 可视化客户端数据分布
    visualize_distribution(client_dist, f"MNIST数据集的狄利克雷划分 (α={alpha})")
    
    return client_datasets, client_dist
