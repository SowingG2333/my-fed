from fed_global import MNIST_Net, Fashion_Net, CIFAR10_AlexNet, data_generate
from adam_FR import AdamFreeRider
from normal_client import NormalClient
from torch.utils.data import DataLoader
import torch

def cosine_similarity(grad1, grad2):
    """计算两个梯度的余弦相似度"""
    flat_grad1 = grad1.flatten()
    flat_grad2 = grad2.flatten()
    dot_product = torch.dot(flat_grad1, flat_grad2)
    norm_product = torch.norm(flat_grad1) * torch.norm(flat_grad2)
    return (dot_product / norm_product).item() if norm_product != 0 else 0.0

def evaluate_model(model, dataloader, criterion, device='cpu'):
    """评估函数"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
    return total_loss / total, correct / total

class BlockChain():
    """区块链类"""
    def __init__(self, num_clients):
        self.log = {i: {'pos':[], 'neg':[]} for i in range(num_clients)}

    def add_pos(self, cid, round_num):
        self.log[cid]['pos'].append(round_num)

    def add_neg(self, cid, round_num):
        self.log[cid]['neg'].append(round_num)

class Commitee():
    """委员会类"""
    def __init__(self, member_num, num_client, blockchain):
        self.member_list = []
        self.reput_dict = {}
        self.member_num = member_num
        self.num_client = num_client
        self.blockchain = blockchain

    def add_member(self, client):
        if(self.member_list.len() < self.member_num):
            self.member_list.append(client)
        else:
            print("委员会已满，无法添加新会员")

    def clear_member(self):
        self.member_list = []

    def reputation_init(self):
        """初始化声誉"""
        self.reput_dict = {i: [0.5, 0.5] for i in range(self.num_client)}

    def cal_reputation(self, cid, round_num, decay_factor):
        """根据主观逻辑权重模型计算声誉"""
        pos_numer = 0
        neg_numer = 0

        for i in range(len(self.blockchain.log[cid]['pos'])):
            pos_numer += decay_factor ^ (round_num - self.blockchain.log[cid]['pos'][i])
            neg_numer += decay_factor ^ (round_num - self.blockchain.log[cid]['neg'][i])
        
        denom = pos_numer + neg_numer

        b = pos_numer / denom if denom != 0 else 0
        d = neg_numer / denom if denom != 0 else 0

        self.reput_dict[cid][0] = b
        self.reput_dict[cid][1] = d

        return b, d
    
    def get_reputation(self, cid):
        """获取声誉"""
        return self.reput_dict[cid][0], self.reput_dict[cid][1]

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 初始化环境
    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    print(f"using {device} to train")

    global_model = Fashion_Net()
    criterion = torch.nn.CrossEntropyLoss() 

    num_clients = 10
    train_per = 0.75
    diri_alpha = 0.5
    # 多生成一个公共数据集
    client_datasets, test_dataset = data_generate(num_clients + 1, train_per, diri_alpha, data_type='FashionMNIST')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    normal_clients = []
    for cid in range(num_clients):
        normal_clients.append(NormalClient(cid,
                                        device, 
                                        global_model, 
                                        lr=0.01,
                                        optimizer='Adam',
                                        criterion=criterion, 
                                        betas=(0.9, 0.999),
                                        eps=1e-8,
                                        batch_size=64, 
                                        local_epochs=1, 
                                        dataset=client_datasets[cid]))

    free_rider = AdamFreeRider(num_clients,
                               device,
                               public_dataset=client_datasets[num_clients],
                               global_model=global_model,
                               criterion=criterion,
                               lr=0.01,
                               batch_size=64,
                               betas=(0.9, 0.999),
                               eps=1e-8,
                               sigma_n=0.1)

    # 修正后的数据存储结构
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

    # 模拟联邦学习
    for round_num in range(10):
        print(f"\n=== Round {round_num} ===")
        round_grads = {'normal': {}, 'free_rider': None}
        client_updates = []
        
        # 正常客户端训练
        for normal_client in normal_clients:
            normal_client.update_global_model(global_model.state_dict())
            new_global_model_state_dict, real_grad = normal_client.local_train()
            print(real_grad)
            client_updates.append(new_global_model_state_dict)
            
            # 存储梯度指标
            cid = normal_client.cid
            last_layer_grad = torch.cat([grad.flatten() for name, grad in real_grad.items() 
                                      if name in free_rider.last_layer_params])
            history['norms']['normal'][cid].append(last_layer_grad.norm().item())
            round_grads['normal'][cid] = last_layer_grad

        # 参数聚合
        aggregated_state_dict = {}
        for param_name in client_updates[0].keys():
            param_list = [client_state[param_name] for client_state in client_updates]
            aggregated_state_dict[param_name] = torch.stack(param_list).float().mean(dim=0)
        global_model.load_state_dict(aggregated_state_dict)
        
        # 模型评估
        test_loss, test_acc = evaluate_model(
            global_model,
            test_loader,
            criterion,
            device=next(global_model.parameters()).device
        )
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        
        # 搭便车者处理
        free_rider.update_global_model(global_model.state_dict())
        fake_grads = free_rider.generate_fake_grad(round_num)
        last_layer_grad = torch.cat([grad.flatten() for name, grad in fake_grads.items()])
        history['norms']['free_rider'].append(last_layer_grad.norm().item())
        round_grads['free_rider'] = last_layer_grad
        
        # 计算并存储相似度
        normal_grads = [grad.clone() for grad in round_grads['normal'].values()]
        
        # 1. 归一化每个正常客户端梯度
        normalized_grads = []
        for grad in normal_grads:
            norm = grad.norm(p=2)
            normalized_grads.append(grad / norm if norm > 0 else grad)
        
        # 2. 计算归一化后的平均梯度
        avg_normal_grad = torch.stack(normalized_grads).mean(dim=0)
        
        # 3. 再次归一化平均梯度
        avg_norm = avg_normal_grad.norm(p=2)
        avg_normal_grad_normalized = avg_normal_grad / avg_norm if avg_norm > 0 else avg_normal_grad
        
        # 4. 处理搭便车者梯度
        free_rider_grad = round_grads['free_rider'].clone()
        fr_norm = free_rider_grad.norm(p=2)
        free_rider_grad_normalized = free_rider_grad / fr_norm if fr_norm > 0 else free_rider_grad

        # 5. 计算余弦相似度
        adam_sim = torch.dot(avg_normal_grad_normalized.to(device), free_rider_grad_normalized.to(device)).item()
        history['similarity']['free_rider'].append(adam_sim)
        cid = 0
        for normal_grad in normalized_grads:
            normal_sim = torch.dot(normal_grad, avg_normal_grad_normalized).item()
            history['similarity']['normal'][cid].append(normal_sim)
            cid += 1

    # 修正后的可视化代码
    plt.figure(figsize=(15, 12))
    
    # 梯度范数趋势
    plt.subplot(3, 2, 1)
    for cid in range(num_clients):
        plt.plot(history['norms']['normal'][cid], 'b-', alpha=0.3)
    # 绘制平均梯度范数
    avg_norms = [np.mean([v[i] for v in history['norms']['normal'].values()]) for i in range(len(history['norms']['normal'][0]))]
    plt.plot(avg_norms, 'g-', linewidth=2)
    plt.plot(history['norms']['free_rider'], 'r--', linewidth=2)
    plt.title('Gradient Norms\n(Blue: Normal Clients, Red: Free Rider)')
    plt.xlabel('Training Round')
    plt.ylabel('Norm')
    
    # 余弦相似度趋势
    plt.subplot(3, 2, 2)
    for cid in range(num_clients):
        plt.plot(history['similarity']['normal'][cid], 'b-', alpha=0.3)
    # 绘制平均相似度
    avg_sims = [np.mean([v[i] for v in history['similarity']['normal'].values()]) for i in range(len(history['similarity']['normal'][0]))]
    plt.plot(avg_sims, 'g-', linewidth=2)
    plt.plot(history['similarity']['free_rider'], 'r--', linewidth=2)
    plt.title('Cosine Similarity to Normalized Average\n(Blue: Normal Clients, Red: Free Rider)')
    plt.xlabel('Training Round')
    plt.ylabel('Similarity')
    
    # 模型性能
    plt.subplot(3, 2, 3)
    plt.plot(history['test_acc'], 'g-', label='Accuracy')
    plt.plot(history['test_loss'], 'm--', label='Loss')
    plt.title('Model Performance')
    plt.xlabel('Training Round')
    plt.legend()
    
    # 修正后的统计信息
    plt.subplot(3, 2, 5)
    stats_text = (
        f"Final Metrics:\n"
        f"Final Similarity: {history['similarity']['free_rider'][-1]:.3f}\n"
        f"Free Rider Norm: {history['norms']['free_rider'][-1]:.2f}\n"
        f"Normal Clients Avg Norm: {np.mean([v[-1] for v in history['norms']['normal'].values()]):.2f}\n"
        f"Test Accuracy: {history['test_acc'][-1]*100:.1f}%"
    )
    plt.text(0.1, 0.2, stats_text, 
            bbox={'facecolor': 'lightyellow', 'alpha': 0.5})
    plt.axis('off')
    
    # 修正后的相似度分布直方图
    plt.subplot(3, 2, 4)
    plt.hist(history['similarity']['free_rider'], bins=20, color='blue', alpha=0.7)
    plt.title('Similarity Distribution\n(All Rounds)')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('training_analysis_with_similarity.png')
    plt.close()

    print("可视化图表已保存至 training_analysis_with_similarity.png")