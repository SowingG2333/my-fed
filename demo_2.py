import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 模型定义 ======================
class FedModel(nn.Module):
    """联邦学习全局模型（修正维度）"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),  # 权重形状应为 [256, 784]
            nn.ReLU(),
            nn.Linear(256, 128),  # 权重形状 [128, 256]
            nn.ReLU(),
            nn.Linear(128, 10)    # 权重形状 [10, 128]
        )
        # 参数初始化
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# ====================== 客户端组件 ======================
class NormalClient:
    """正常客户端训练器（保持维度一致）"""
    def __init__(self, cid, global_model, lr, batch_size, local_epochs, dataset):
        self.cid = cid
        self.global_model = copy.deepcopy(global_model)
        self.local_model = copy.deepcopy(global_model)
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.local_epochs = local_epochs
        self.data_size = len(dataset)

    def local_train(self):
        """执行本地训练并返回梯度更新（保持参数形状）"""
        initial_state = copy.deepcopy(self.local_model.state_dict())
        
        self.local_model.train()
        for _ in range(self.local_epochs):
            for X, y in self.loader:
                self.optimizer.zero_grad()
                pred = self.local_model(X)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
        
        return {
            name: (initial_state[name] - self.local_model.state_dict()[name])
            for name in initial_state.keys()
        }

class AdamFreeRider:
    def __init__(self, cid, global_model, lr, adam_params):
        self.cid = cid
        self.model = copy.deepcopy(global_model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=adam_params)
        self.alpha, self.beta = adam_params
        # 预初始化所有参数的delta_0
        self.delta_0 = {name: None for name, _ in self.model.named_parameters()}

    def generate_fake_grad(self, global_model, round_num):
        """修复初始化逻辑"""
        self.model.load_state_dict(global_model.state_dict())
        fake_grad = {}
        
        # 严格遍历所有参数
        for name, param in self.model.named_parameters():
            if round_num == 0:
                if self.delta_0[name] is None:
                    noise = torch.randn_like(param)
                    self.delta_0[name] = noise  # 确保存储到正确键名
                    print(f"初始化参数: {name}")  # 调试输出
                else:
                    raise RuntimeError(f"参数 {name} 重复初始化!")
            else:
                if self.delta_0[name] is None:
                    raise RuntimeError(f"参数 {name} 未在首轮初始化!")

                # 生成混合噪声
                base = self.alpha * self.delta_0[name]
                new_noise = self.beta * torch.randn_like(param)
                total_noise = base + new_noise
                param.grad = total_noise
                fake_grad[name] = total_noise.detach()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self.model.state_dict(), fake_grad

# ====================== 防御机制 ======================
class CosineDefender:
    """基于余弦相似度的搭便车攻击检测器（维度校验）"""
    def __init__(self, max_num, cos_threshold, model):
        self.max_num = max_num
        self.cos_threshold = cos_threshold
        self.model = model
        self.reset()
    
    def reset(self):
        self.c_num = 0
        self.c_ids = []
        self.avg_grad = {}
        self.gradients = []

    def add_update(self, cid, grad_update):
        """添加客户端更新（维度校验）"""
        # 参数维度校验
        for name, grad in grad_update.items():
            expected_shape = self.model.state_dict()[name].shape
            if grad.shape != expected_shape:
                raise ValueError(f"参数{name}维度错误: {grad.shape} 应该为 {expected_shape}")
        
        if self.c_num < self.max_num:
            self.c_ids.append(cid)
            for name, grad in grad_update.items():
                if name in self.avg_grad:
                    self.avg_grad[name] += grad
                else:
                    self.avg_grad[name] = grad.clone()
            self.gradients.append(grad_update)
            self.c_num += 1
            return None
        else:
            detected = self.detect()
            self.reset()
            return detected

    def detect(self):
        """执行检测（添加维度校验）"""
        avg_vector = torch.cat([v.flatten() / self.c_num 
                              for v in self.avg_grad.values()]).detach()
        
        malicious = []
        for i, grad in enumerate(self.gradients):
            client_vector = torch.cat([v.flatten() for v in grad.values()]).detach()
            similarity = F.cosine_similarity(avg_vector.unsqueeze(0), 
                                           client_vector.unsqueeze(0),
                                           dim=1).item()
            if similarity < self.cos_threshold:
                malicious.append((self.c_ids[i], similarity))
        
        return malicious

# ====================== 数据工具 ======================
def non_iid_split(dataset, num_clients, classes_per_client=2):
    """非独立同分布数据划分"""
    labels = np.array(dataset.targets.numpy())
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

# ====================== 主流程 ======================
def main():
    # 实验参数
    NUM_CLIENTS = 20
    MALICIOUS_RATIO = 0.3
    NUM_ROUNDS = 30
    CHOOSE_PERCENTAGE = 0.5
    COS_THRESHOLD = 0.4
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 64
    MODEL_LR = 0.01
    ATTACK_PARAMS = (0.7, 0.3)
    
    # 固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 准备数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    client_datasets = non_iid_split(train_set, NUM_CLIENTS)
    test_loader = DataLoader(test_set, batch_size=512)

    # 初始化组件
    global_model = FedModel()
    defender = CosineDefender(
        max_num=int(NUM_CLIENTS * CHOOSE_PERCENTAGE),
        cos_threshold=COS_THRESHOLD,
        model=global_model
    )
    
    # 创建客户端池（添加维度校验）
    all_clients = list(range(NUM_CLIENTS))
    malicious_ids = np.random.choice(all_clients, 
                                    size=int(NUM_CLIENTS * MALICIOUS_RATIO),
                                    replace=False).tolist()
    
    client_pool = {}
    for cid in all_clients:
        if cid in malicious_ids:
            client_pool[cid] = {
                'type': 'malicious',
                'handler': AdamFreeRider(cid, global_model, MODEL_LR, ATTACK_PARAMS)
            }
        else:
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

    # 训练记录
    history = {
        'test_acc': [],
        'detected': [],
        'similarities': [],
        'client_types': []
    }

    # 联邦训练循环（修正聚合逻辑）
    for round_num in range(NUM_ROUNDS):
        # 选择参与客户端
        selected = np.random.choice(all_clients, 
                                   size=int(NUM_CLIENTS * CHOOSE_PERCENTAGE),
                                   replace=False)
        
        # 收集更新（添加维度校验）
        all_grads = []
        round_types = []
        for cid in selected:
            client = client_pool[cid]
            try:
                if client['type'] == 'malicious':
                    _, grad = client['handler'].generate_fake_grad(global_model, round_num)
                    round_types.append(1)
                else:
                    grad = client['handler'].local_train()
                    round_types.append(0)
                all_grads.append(grad)
                defender.add_update(cid, grad)
            except ValueError as e:
                print(f"客户端{cid}梯度异常: {str(e)}")
                continue

        # 执行防御检测
        detection_result = defender.add_update(None, {})  # 触发检测
        if detection_result is not None:
            detected_ids = [cid for cid, _ in detection_result]
            true_labels = [1 if c in malicious_ids else 0 for c in selected]
            pred_labels = [1 if c in detected_ids else 0 for c in selected]
            
            if sum(true_labels) > 0:
                precision = precision_score(true_labels, pred_labels)
                recall = recall_score(true_labels, pred_labels)
                history['detected'].append({
                    'round': round,
                    'precision': precision,
                    'recall': recall,
                    'detected': detected_ids
                })
                history['similarities'].extend([score for _, score in detection_result])
                history['client_types'].extend(round_types)

        # 聚合更新（修正维度处理）
        valid_grads = [g for cid, g in zip(selected, all_grads) 
                      if cid not in (detected_ids if 'detected_ids' in locals() else [])]
        if valid_grads:
            # 数据量加权平均（保持正确维度）
            weights = [client_pool[cid]['handler'].data_size 
                      if client_pool[cid]['type'] == 'normal' else 0
                      for cid in selected]
            valid_weights = [w for cid, w in zip(selected, weights)
                            if cid not in (detected_ids if 'detected_ids' in locals() else [])]
            total_weight = sum(valid_weights)
            
            if total_weight > 0:
                aggregated = {}
                for name in global_model.state_dict().keys():
                    # 堆叠梯度并保持正确形状 [num_clients, *param_shape]
                    param_grads = torch.stack([g[name] for g in valid_grads])
                    # 计算加权平均 [param_shape]
                    weighted_avg = (param_grads * torch.tensor(valid_weights)[:, None, None] / total_weight).sum(dim=0)
                    aggregated[name] = global_model.state_dict()[name] + weighted_avg
                
                # 加载聚合参数（严格形状校验）
                try:
                    global_model.load_state_dict(aggregated)
                except RuntimeError as e:
                    print(f"聚合参数形状错误: {str(e)}")
                    continue

        # 模型评估
        global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                outputs = global_model(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        history['test_acc'].append(correct / total)

        # 打印进度
        print(f"Round {round_num+1}/{NUM_ROUNDS} | 准确率: {history['test_acc'][-1]:.2%} | 检测到恶意: {len(detected_ids) if 'detected_ids' in locals() else 0}")

    # 保存结果
    with open('fl_log.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 可视化
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(history['test_acc'], label='测试准确率')
    plt.title('模型性能')
    
    plt.subplot(132)
    plt.hist([s for s, t in zip(history['similarities'], history['client_types']) if t == 1],
             bins=20, alpha=0.7, label='恶意客户端')
    plt.hist([s for s, t in zip(history['similarities'], history['client_types']) if t == 0],
             bins=20, alpha=0.7, label='正常客户端')
    plt.title('梯度相似度分布')
    plt.legend()
    
    plt.subplot(133)
    precisions = [d['precision'] for d in history['detected']]
    recalls = [d['recall'] for d in history['detected']]
    plt.plot(precisions, label='查准率')
    plt.plot(recalls, label='查全率')
    plt.title('防御性能')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('fl_result.png')
    plt.show()

if __name__ == "__main__":
    main()