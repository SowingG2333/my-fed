import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class AdamFreeRider:
    """实现联邦学习中对抗性梯度生成的优化器"""
    def __init__(self, cid, global_model, lr, adam_params):
        self.cid = cid
        self.model = copy.deepcopy(global_model)
        self.optimzier = optim.Adam(self.model.parameters(), lr, adam_params)
        self.delta_0 = {}
        self.alpha, self.beta = adam_params[0], adam_params[1]

    def generate_fake_grad(self, round_num, global_model):
        """生成虚假梯度更新，并返回更新后的模型参数与更新梯度"""
        # 加载全局模型
        self.model.load_state_dict(global_model.state_dict())
        # 生成噪声梯度
        update_dict = self.adaptive_perturbation(round_num, self.alpha, self.beta)

        return self.model.state_dict(), update_dict

    def adaptive_perturbation(self, round_num, alpha, beta):
        """生成根据adam优化器公式的梯度更新"""
        fake_gradients = {}

        for name, param in self.model.named_parameters():
            if round_num == 0:
                # 生成首轮噪声梯度
                noise = torch.rand_like(param)
                param.grad = noise
                fake_gradients[name] = noise
                self.delta_0 = fake_gradients

            else:
            # 生成融合梯度
                base_grad = self.delta_0[name] * alpha
                noise = torch.randn_like(param) * beta
                fake_grad = base_grad + noise

                # 梯度赋值
                param.grad = fake_grad
                fake_gradients[name] = fake_grad

        # 执行Adam优化器更新
        self.optimzier.step()
        self.optimzier.zero_grad()

        return fake_gradients

class NormalClient:
    """正常客户端训练器"""
    def __init__(self, cid, global_model, lr, batch_size, local_epochs, dataset):
        self.cid = cid
        self.global_model = global_model
        self.local_model = copy.deepcopy(global_model)
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.local_epochs = local_epochs

    def local_train(self):
        """执行本地训练并返回梯度更新"""
        # 保留初始参数
        initial_state = copy.deepcopy(self.local_model.state_dict())
        
        # 训练过程
        self.local_model.train()
        for _ in range(self.local_epochs):
            for X, y in self.loader:
                self.optimizer.zero_grad()
                pred = self.local_model(X)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
        
        # 计算梯度更新（当前参数与初始参数的差值）
        grad_update = {
            name: initial_state[name] - self.local_model.state_dict()[name] 
            for name in initial_state.keys()
        }
        return grad_update

class CosineDefender:
    """基于余弦相似度计算的搭便车攻击防御器"""
    def __init__(self, max_num, cos_threshold, model):
        self.c_num = 0                       # 当前累积的客户端数量
        self.max_num = max_num               # 触发检测的客户端数量阈值
        self.c_ids = []                      # 储存客户端id（用于检测时的映射）
        self.avg_gradients = {}              # 梯度累积字典
        self.gradients = []                  # 存储所有客户端梯度
        self.cos_threshold = cos_threshold   # 相似度阈值
        self.model = model                   # 参考模型（用于参数结构）

    def FR_detection(self):
        """检测恶意客户端，返回恶意客户端索引列表"""
        # 计算平均梯度向量
        avg_grad = {name: grad / self.c_num for name, grad in self.avg_gradients.items()}
        avg_vector = torch.cat([v.flatten() for v in avg_grad.values()]).detach()

        # 检测低相似度客户端
        malicious = []
        for i, client_grad in enumerate(self.gradients):
            # 展平客户端梯度
            client_vector = torch.cat([v.flatten() for v in client_grad.values()]).detach()
            
            # 计算余弦相似度
            similarity = F.cosine_similarity(avg_vector.unsqueeze(0), 
                                             client_vector.unsqueeze(0), 
                                             dim=1).item()
            
            if similarity < self.cos_threshold:
                malicious.append((self.c_ids[i], similarity))
        
        return malicious    # 返回形式 [(id, 相似度分数)...]

    def add_update(self, client_id, client_grads):
        """添加客户端梯度更新"""
        if self.c_num <= self.max_num:
            self.c_ids.append(client_id)
            # 存储梯度并累加
            self.gradients.append(client_grads)
            for name, grad in client_grads.items():
                if name not in self.avg_gradients:
                    self.avg_gradients[name] = grad.clone()
                else:
                    self.avg_gradients[name] += grad
            self.c_num += 1
            return None
        else:
            # 触发检测并重置
            malicious_clients = self.FR_detection()
            print(f"触发异步聚合，检测到恶意客户端索引: {malicious_clients}")
            self.clear_avg_grad()
            self.c_num = 0
            self.gradients = []
            return malicious_clients  # 返回恶意客户端列表

    def clear_avg_grad(self):
        """清空累积梯度和id"""
        self.avg_gradients = {}
        self.c_ids = []

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

def main():
    NUM_CLIENTS = 20                 # 客户端总数
    MALICIOUS_RATIO = 0.1            # 恶意客户端比例
    NUM_ROUNDS = 30                  # 联邦训练轮次
    CHOOSE_PERCENTAGE = 0.5          # 每轮学习参与率
    COS_THRESHOLD = 0.5              # 防御阈值
    LOCAL_EPOCHS = 10                # 本地训练轮次
    BATCH_SIZE = 64                  # 本地训练批次大小
    MODEL_LR = 0.1                   # 模型学习率
    ATTACK_PARAMS = (0.8, 0.1)       # 攻击参数(α, β)

    # MNIST数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载完整数据集
    full_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 创建客户端数据集
    client_datasets = non_iid_split(full_dataset, NUM_CLIENTS)

    # 训练记录器
    history = {
        'similarities': [],
        'precision': [],
        'recall': [],
        'detected': []
    }

    # 初始化全局模型和防御器
    global_model = FedModel()
    defender = CosineDefender(cos_threshold=COS_THRESHOLD,
                              max_num=NUM_CLIENTS * CHOOSE_PERCENTAGE,
                              model=global_model)
    
    # 选择恶意客户端
    all_clients = list(range(NUM_CLIENTS))
    malicious_clients = np.random.choice(all_clients, 
                                         size=int(NUM_CLIENTS * MALICIOUS_RATIO),
                                         replace=False).tolist()
    
    # 创建所有客户端实例
    client_pool = {}
    for cid in range(NUM_CLIENTS):
        if cid in malicious_clients:
            # 恶意客户端实例
            client_pool[cid] = {
                'type': 'malicious',
                'handler': AdamFreeRider(cid, global_model, MODEL_LR, ATTACK_PARAMS)
            }
        else:
            # 正常客户端实例
            client_pool[cid] = {
                'type': 'normal',
                'handler': NormalClient(
                    cid=cid,
                    global_model=global_model,
                    lr=MODEL_LR,
                    batch_size=BATCH_SIZE,
                    local_epochs=LOCAL_EPOCHS,
                    dataset=client_datasets[cid]
                )
            }

    # 联邦训练循环
    for round_num in range(NUM_ROUNDS):
        print(f"round {round_num}")
        # 选择参与客户端
        selected_clients = np.random.choice(
            all_clients, 
            size=int(NUM_CLIENTS * CHOOSE_PERCENTAGE),
            replace=False
        )
        
        # 收集梯度更新
        all_gradients = []
        for cid in selected_clients:
            client = client_pool[cid]
            
            if client['type'] == 'malicious':
                # 生成恶意梯度
                _, grad = client['handler'].generate_fake_grad(round_num, global_model)
            else:
                # 执行正常训练
                grad = client['handler'].local_train()
            
            all_gradients.append(grad)
            defender.add_update(cid, grad)

        # 执行防御检测
        detected_malicious = defender.FR_detection()
        detected_ids = [cid for (cid, _) in detected_malicious]

        # 记录检测结果
        history['detected'].append({
            'round': round,
            'malicious': detected_ids,
            'scores': [score for (_, score) in detected_malicious]
        })

        # 收集有效客户端的梯度及其数据量权重
        valid_grads = []
        data_weights = []
        
        for cid, grad in zip(selected_clients, all_gradients):
            if cid not in detected_ids:
                valid_grads.append(grad)
                # 获取数据量权重（正常客户端从数据集获取，恶意客户端设为0）
                if client_pool[cid]['type'] == 'normal':
                    data_size = len(client_pool[cid]['handler'].loader.dataset)
                else:
                    data_size = 0  # 恶意客户端即使没被检测也视为无效
                data_weights.append(data_size)

        # 标准化权重
        total_weight = sum(data_weights)
        if total_weight == 0:
            print("警告：无有效客户端，跳过本轮聚合")
            continue
            
        normalized_weights = [w/total_weight for w in data_weights]

        # 加权聚合
        aggregated = {}
        for param_name in global_model.state_dict().keys():
            # 堆叠所有客户端的对应参数梯度
            param_grads = torch.stack([g[param_name] for g in valid_grads])
            
            # 计算加权平均
            weighted_avg = torch.zeros_like(param_grads[0])
            for grad, weight in zip(param_grads, normalized_weights):
                weighted_avg.add_(grad.to(weighted_avg.device) * weight)
            
            aggregated[param_name] = global_model.state_dict()[param_name] + weighted_avg

        # 更新全局模型
        global_model.load_state_dict(aggregated)

main()