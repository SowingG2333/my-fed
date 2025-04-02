import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from fed_global import FedModel, data_generate
from adam_FR import AdamFreeRider
from normal_client import NormalClient

def evaluate_model(model, dataloader, criterion, device=None):
    model.eval()
    device = device or next(model.parameters()).device
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / total, correct / total

def track_similarities(history, round_grads):
    device = next(global_model.parameters()).device
    
    # 梯度归一化处理
    def normalize(grad):
        norm = grad.norm(p=2).to(device)
        return grad / norm if norm > 0 else grad
    
    # 处理正常客户端梯度
    normal_grads = [normalize(grad) for grad in round_grads['normal'].values()]
    avg_normal = torch.stack(normal_grads).mean(dim=0).to(device)
    avg_normal = normalize(avg_normal)
    
    # 记录相似度
    history['similarity']['free_rider'].append(
        torch.dot(avg_normal, normalize(round_grads['free_rider'])).item()
    )
    
    for cid, grad in enumerate(normal_grads):
        history['similarity']['normal'][cid].append(
            torch.dot(avg_normal, normalize(grad)).item()
        )
# ======================
# CUDA加速核心配置
# ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ======================
# 模型初始化（带GPU支持）
# ======================
global_model = FedModel().to(device)
criterion = nn.CrossEntropyLoss().to(device)

# ======================
# 优化器配置（可选混合精度）
# ======================
try:
    from apex import amp
    optimizer = torch.optim.AdamW(global_model.parameters(), lr=0.01)
    global_model, optimizer = amp.initialize(
        global_model, optimizer, opt_level="O1"
    )
except ImportError:
    optimizer = torch.optim.AdamW(global_model.parameters(), lr=0.01)

# ======================
# 客户端初始化
# ======================
num_clients = 5
diri_alpha = 0.9

# 数据生成（添加设备参数）
client_datasets, test_dataset = data_generate(
    num_clients, diri_alpha, data_type='CIFAR10', device=device
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=64, 
    shuffle=False,
    num_workers=4,       # 多线程数据加载
    pin_memory=True      # 内存锁定加速传输
)

normal_clients = [
    NormalClient(
        cid=cid, 
        global_model=global_model,
        lr=0.01, 
        betas=(0.9, 0.999),
        eps=1e-8,
        batch_size=64, 
        local_epochs=10, 
        dataset=client_datasets[cid],
        device=device  # 传递设备参数
    ) for cid in range(num_clients)
]

# ======================
# 梯度跟踪容器
# ======================
history = {
    'norms': {
        'normal': {cid: [] for cid in range(num_clients)},
        'free_rider': []
    },
    'similarity': {
        'normal': {cid: [] for cid in range(num_clients)},
        'free_rider': []
    },
    'test_acc': [],
    'test_loss': []
}

# ======================
# 主训练循环（GPU加速版）
# ======================
for round_num in range(50):
    print(f"
=== Round {round_num} ===")
    round_grads = {'normal': {}, 'free_rider': None}
    client_updates = []
    
    # 正常客户端训练
    for client in normal_clients:
        client.update_global_model(global_model.state_dict())
        new_state, real_grad = client.local_train()
        
        # 梯度处理（自动在GPU执行）
        cid = client.cid
        all_grads = torch.cat([g.flatten() for g in real_grad.values()])
        history['norms']['normal'][cid].append(all_grads.norm().item())
        round_grads['normal'][cid] = all_grads
        
        client_updates.append(new_state)
    
    # 参数聚合
    aggregated = {}
    for name in client_updates[0]:
        aggregated[name] = torch.stack([c[name] for c in client_updates]).mean(dim=0)
    global_model.load_state_dict(aggregated)
    
    # 模型评估
    test_loss, test_acc = evaluate_model(
        global_model, 
        test_loader, 
        criterion, 
        device=device
    )
    history['test_acc'].append(test_acc)
    history['test_loss'].append(test_loss)
    
    # 搭便车者处理
    free_rider = AdamFreeRider(
        cid=0,
        global_model=global_model,
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        sigma_n=0.1
    ).to(device)
    
    free_rider.update_global_model(global_model.state_dict())
    fake_grads = free_rider.generate_fake_grad(round_num)
    round_grads['free_rider'] = fake_grads
    
    # 相似度计算（全GPU计算）
    track_similarities(history, round_grads)

# ======================
# 可视化输出
# ======================
plt.figure(figsize=(18, 12))

# 梯度范数趋势
plt.subplot(3, 2, 1)
for cid in history['norms']['normal']:
    plt.plot(history['norms']['normal'][cid], 'b-', alpha=0.3)
plt.plot(history['norms']['free_rider'], 'r--', linewidth=2)
plt.title('Gradient Norms')
plt.xlabel('Round')
plt.ylabel('Norm')

# 相似度分布
plt.subplot(3, 2, 2)
for cid in history['similarity']['normal']:
    plt.plot(history['similarity']['normal'][cid], 'b-', alpha=0.3)
plt.plot(history['similarity']['free_rider'], 'r--', linewidth=2)
plt.title('Cosine Similarity')
plt.xlabel('Round')
plt.ylabel('Similarity')

# 性能曲线
plt.subplot(3, 2, 3)
plt.plot(history['test_acc'], 'g-', label='Accuracy')
plt.plot(history['test_loss'], 'm--', label='Loss')
plt.legend()
plt.title('Model Performance')

# 统计信息
plt.subplot(3, 2, 4)
stats = (
    f"Final Metrics:
"
    f"- Free Rider Norm: {history['norms']['free_rider'][-1]:.2f}
"
    f"- Avg Normal Norm: {np.mean(list(history['norms']['normal'].values())):.2f}
"
    f"- Test Acc: {history['test_acc'][-1]*100:.1f}%"
)
plt.text(0.1, 0.5, stats, fontsize=10)
plt.axis('off')

# 直方图分布
plt.subplot(3, 2, 5)
plt.hist(history['similarity']['free_rider'], bins=20, color='royalblue')
plt.title('Similarity Distribution')

plt.tight_layout()
plt.savefig('gpu_optimized_analysis.png')
plt.close()

print("GPU加速完成！可视化保存为 gpu_optimized_analysis.png")
# ======================
# 数据生成器修改（添加设备支持）
# ======================
def data_generate(num_clients, diri_alpha, data_type='CIFAR10', device=None):
    # 假设原始数据生成逻辑...
    datasets = []
    
    for _ in range(num_clients):
        # 生成数据集...
        dataset = ...  # 原始数据生成逻辑
        
        # 添加设备迁移
        if device:
            dataset.data = dataset.data.to(device)
            dataset.targets = dataset.targets.to(device)
        
        datasets.append(dataset)
    
    return datasets, test_dataset