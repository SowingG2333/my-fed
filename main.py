from fed_global import FedModel, data_generate
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

import matplotlib.pyplot as plt
import numpy as np

def track_similarities(history, round_grads):
    """记录搭便车者与归一化后的真实客户端平均梯度的余弦相似度"""
    # 获取所有正常客户端的梯度
    normal_grads = [grad.clone() for grad in round_grads['normal'].values()]
    
    # 跳过空梯度的情况
    if len(normal_grads) == 0:
        return
    
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
    adam_sim = torch.dot(avg_normal_grad_normalized, free_rider_grad_normalized).item()
    history['similarity']['free_rider'].append(adam_sim)
    cid = 0
    for normal_grad in normalized_grads:
        normal_sim = torch.dot(normal_grad, avg_normal_grad_normalized).item()
        history['similarity']['normal'][cid].append(normal_sim)
        cid += 1

if __name__ == '__main__':
    # 初始化环境
    from fed_global import FedModel
    from normal_client import NormalClient

    global_model = FedModel()
    criterion = torch.nn.CrossEntropyLoss() 
    free_rider = AdamFreeRider(
        cid=0,
        global_model=global_model,
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        sigma_n=0.1
    )
    normal_clients = []

    num_clients = 5
    diri_alpha = 0.1
    client_datasets, test_dataset = data_generate(num_clients, diri_alpha)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    for cid in range(num_clients):
        normal_clients.append(NormalClient(cid, 
                                        global_model, 
                                        lr=0.01, 
                                        betas=(0.9, 0.999),
                                        eps=1e-8,
                                        batch_size=64, 
                                        local_epochs=1, 
                                        dataset=client_datasets[cid]))
    
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
    for round_num in range(50):
        print(f"\n=== Round {round_num} ===")
        round_grads = {'normal': {}, 'free_rider': None}
        client_updates = []
        
        # 正常客户端训练
        for normal_client in normal_clients:
            normal_client.update_global_model(global_model.state_dict())
            new_global_model_state_dict, real_grad = normal_client.local_train()
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
            aggregated_state_dict[param_name] = torch.stack(param_list).mean(dim=0)
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
        track_similarities(history, round_grads)

    # 修正后的可视化代码
    plt.figure(figsize=(15, 12))
    
    # 梯度范数趋势
    plt.subplot(3, 2, 1)
    for cid in range(num_clients):
        plt.plot(history['norms']['normal'][cid], 'b-', alpha=0.3)
    plt.plot(history['norms']['free_rider'], 'r--', linewidth=2)
    plt.title('Gradient Norms\n(Blue: Normal Clients, Red: Free Rider)')
    plt.xlabel('Training Round')
    plt.ylabel('Norm')
    
    # 余弦相似度趋势
    plt.subplot(3, 2, 2)
    for cid in range(num_clients):
        plt.plot(history['similarity']['normal'][cid], 'b-', alpha=0.3)
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

# if __name__ == '__main__':
#     # 初始化环境
#     global_model = FedModel()
#     criterion = torch.nn.CrossEntropyLoss() 
#     free_rider = AdamFreeRider(
#         cid=0,
#         global_model=global_model,
#         lr=0.01,
#         betas=(0.9, 0.999),
#         eps=1e-8,
#         sigma_n=0.5
#     )
#     normal_clients = []

#     num_clients = 5
#     diri_alpha = 0.1
#     client_datasets, test_dataset = data_generate(num_clients, diri_alpha)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#     for cid in range(num_clients):
#         normal_clients.append(NormalClient(cid, 
#                                            global_model, 
#                                            lr=0.01, 
#                                            betas=(0.9, 0.999),
#                                            eps=1e-8,
#                                            batch_size=64, 
#                                            local_epochs=10, 
#                                            dataset=client_datasets[cid]))
#     # 模拟联邦学习
#     for round_num in range(50):
#         print(f"\n=== Round {round_num} ===")
#         real_grads = []
#         client_updates = []
        
#         # 正常客户端训练
#         for normal_client in normal_clients:
#             normal_client.update_global_model(global_model.state_dict())
#             new_global_model_state_dict, real_grad = normal_client.local_train()
#             client_updates.append(new_global_model_state_dict)
#             real_grads.append(real_grad)
        
#         # 参数聚合
#         aggregated_state_dict = {}
#         for param_name in client_updates[0].keys():
#             param_list = [client_state[param_name] for client_state in client_updates]
#             aggregated_state_dict[param_name] = torch.stack(param_list).mean(dim=0)
        
#         # 更新全局模型
#         global_model.load_state_dict(aggregated_state_dict)
        
#         # 新增模型评估
#         test_loss, test_acc = evaluate_model(
#             global_model,
#             test_loader,
#             criterion,
#             device=next(global_model.parameters()).device
#         )
#         print(f"\n[Global Model Evaluation]")
#         print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")
        
#         # 搭便车者更新和梯度生成
#         free_rider.update_global_model(global_model.state_dict())
#         fake_grads = free_rider.generate_fake_grad(round_num)
        
#         # 计算余弦相似度（保留原有逻辑）
#         total_sim = 0.0
#         valid_params = 0
        
#         # 计算真实梯度的平均值
#         avg_real_grads = {}
#         for real_grad in real_grads:
#             for name, grad in real_grad.items():
#                 if name not in avg_real_grads:
#                     avg_real_grads[name] = torch.zeros_like(grad)
#                 avg_real_grads[name] += grad.clone()
#         for name in avg_real_grads:
#             avg_real_grads[name] = avg_real_grads[name] / len(real_grads)

#         # 计算真实梯度的标准差（跨客户端维度）
#         real_grad_stds = {}
#         for name in avg_real_grads:
#             # 收集所有客户端的梯度
#             all_grads = [client_grad[name] for client_grad in real_grads]
#             # 沿着客户端维度计算标准差
#             real_grad_stds[name] = torch.stack(all_grads).std(dim=0)

#         # 计算虚假梯度的标准差
#         fake_grad_stds = {name: torch.std(grad) for name, grad in fake_grads.items()}

        
#         # 遍历所有参数比较梯度
#         for param_name in fake_grads:
#             if param_name in avg_real_grads:
#                 # 确保设备一致
#                 avg_real_grad = avg_real_grads[param_name].to(fake_grads[param_name].device)
#                 fake_grad = fake_grads[param_name]
#                 real_std = real_grad_stds.get(param_name, torch.tensor(0.0))
#                 fake_std = fake_grad_stds.get(param_name, torch.tensor(0.0))
                
#                 sim = cosine_similarity(avg_real_grad, fake_grad)
#                 total_sim += sim
#                 valid_params += 1
#                 print(f"[{param_name}]")
#                 print(f"  Cosine Similarity: {sim:.4f}")
#                 print(f"  Real Grad Norm: {avg_real_grad.norm().item():.4f}")
                
#                 if real_std.numel() > 1:
#                     real_std_print = real_std.mean().item()
#                 else:
#                     real_std_print = real_std.item()
#                 print(f"  Real Grad Std: {real_std_print:.4f}") 
#                 print(f"  Fake Grad Norm: {fake_grad.norm().item():.4f}")

#                 if fake_std.numel() > 1:
#                     fake_std_print = fake_std.mean().item()
#                 else:
#                     fake_std_print = fake_std.item()
#                 print(f"  Fake Grad Std: {fake_std.item():.4f}")    # 新增标准差输出

#         # 输出平均相似度
#         if valid_params > 0:
#             avg_sim = total_sim / valid_params
#             print(f"\nAverage Cosine Similarity: {avg_sim:.4f}")
#         else:
#             print("\nNo matching parameters for comparison")