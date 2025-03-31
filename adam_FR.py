import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

class AdamFreeRiderSimple:
    """模拟联邦学习中基于Adam优化器 每轮输入随机噪声梯度并返回最后一层参数的搭便车节点"""
    def __init__(self, cid, global_model, lr, adam_params):
        self.cid = cid
        self.model = copy.deepcopy(global_model)
        
        # 获取所有参数的名称并确定最后一层的前缀
        param_names = [name for name, _ in self.model.named_parameters()]
        last_param_name = param_names[-1]
        self.last_layer_prefix = last_param_name.rsplit('.', 1)[0]
        self.last_layer_params = [name for name in param_names if name.startswith(self.last_layer_prefix)]
        
        # 将最后一层参数传递给优化器
        last_layer_parameters = [param for name, param in self.model.named_parameters() 
                                 if name in self.last_layer_params]
        self.optimizer = optim.Adam(last_layer_parameters, lr, adam_params)
        
        self.alpha, self.beta = adam_params[0], adam_params[1]
        self.fake_gradients = {}

    def generate_fake_grad(self, round_num):
        """生成针对最后一层参数的伪造梯度更新"""
        current_gradients = {}
        
        for name, param in self.model.named_parameters():
            if name in self.last_layer_params:
                if round_num == 0:
                    # 首轮生成随机噪声作为梯度
                    noise = torch.rand_like(param)
                    param.grad = noise
                    current_gradients[name] = noise
                else:
                    # 后续轮次生成带权重的组合噪声
                    base_grad = self.fake_gradients[name] * self.alpha
                    noise = torch.randn_like(param) * self.beta
                    fake_grad = base_grad + noise
                    param.grad = noise
                    current_gradients[name] = fake_grad
            else:
                # 非最后一层参数不进行梯度更新
                param.grad = None

        # 执行优化器更新
        self.optimizer.step()
        self.optimizer.zero_grad()

        # 更新并返回仅包含最后一层的伪造梯度
        self.fake_gradients.update(current_gradients)
        return self.fake_gradients

class AdamFreeRider:
    """模拟联邦学习中基于Adam优化器 每轮全局模型更新梯度并返回最后一层参数的搭便车节点"""
    def __init__(self, cid, global_model, lr=0.1, betas=(0.9, 0.999), eps=1e-8, sigma_n=0.1):
        self.cid = cid
        self.global_model = copy.deepcopy(global_model)
        self.local_model = copy.deepcopy(global_model)
        self.sigma_n = sigma_n

        # 保存前一轮次的本地参数
        self.prev_local_state = None
        
        # 识别最后一层参数
        param_names = [name for name, _ in self.local_model.named_parameters()]
        last_param_name = param_names[-1]
        self.last_layer_prefix = last_param_name.rsplit('.', 1)[0]
        self.last_layer_params = [name for name in param_names if name.startswith(self.last_layer_prefix)]
        
        # 初始化Adam优化器
        self.optimizer = optim.Adam(
            self.local_model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps
        )
        
    def update_global_model(self, new_global_state):
        """传入新的全局模型 更新本地模型参数"""
        # 保存当前参数用于后续梯度计算
        self.prev_local_state = copy.deepcopy(self.local_model.state_dict())
        
        # 将全局模型参数复制到本地
        local_state = self.local_model.state_dict()
        for key in new_global_state:
            if key in local_state:
                local_state[key].data.copy_(new_global_state[key])

    def generate_fake_grad(self, round_num):
        """生成虚假梯度"""
        original_state = copy.deepcopy(self.local_model.state_dict())
        fake_gradients = {}

        # 生成虚假梯度
        for name, param in self.local_model.named_parameters():
            if round_num == 0:
                noise = torch.normal(mean=0, std=self.sigma_n, size=param.size())
                fake_grad = noise
            else:
                prev_param = self.prev_local_state[name].to(param.device)
                fake_grad = param.data - prev_param

            fake_gradients[name] = fake_grad

        # 应用梯度并更新参数
        self.optimizer.zero_grad()
        for name, param in self.local_model.named_parameters():
            if name in fake_gradients:
                param.grad = fake_gradients[name]
        self.optimizer.step()

        # 提取最后一层实际更新量
        updated_state = self.local_model.state_dict()
        adam_gradients = {
            name: (updated_state[name] - original_state[name]).cpu().clone()
            for name in self.last_layer_params
        }
        
        return adam_gradients

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

from fed_global import data_generate

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
        sigma_n=0.5
    )
    normal_clients = []

    num_clients = 5
    diri_alpha = 0.5
    client_datasets, test_dataset = data_generate(num_clients, diri_alpha)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    for cid in range(num_clients):
        normal_clients.append(NormalClient(cid, 
                                           global_model, 
                                           lr=0.1, 
                                           batch_size=64, 
                                           local_epochs=10, 
                                           dataset=client_datasets[cid]))
    # 模拟联邦学习
    for round_num in range(50):
        print(f"\n=== Round {round_num} ===")
        real_grads = []
        client_updates = []
        
        # 正常客户端训练
        for normal_client in normal_clients:
            normal_client.update_global_model(global_model.state_dict())
            new_global_model_state_dict, real_grad = normal_client.local_train()
            client_updates.append(new_global_model_state_dict)
            real_grads.append(real_grad)
        
        # 参数聚合
        aggregated_state_dict = {}
        for param_name in client_updates[0].keys():
            param_list = [client_state[param_name] for client_state in client_updates]
            aggregated_state_dict[param_name] = torch.stack(param_list).mean(dim=0)
        
        # 更新全局模型
        global_model.load_state_dict(aggregated_state_dict)
        
        # 新增模型评估
        test_loss, test_acc = evaluate_model(
            global_model,
            test_loader,
            criterion,
            device=next(global_model.parameters()).device
        )
        print(f"\n[Global Model Evaluation]")
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")
        
        # 搭便车者更新和梯度生成...
        free_rider.update_global_model(global_model.state_dict())
        fake_grads = free_rider.generate_fake_grad(round_num)
        
        # 计算余弦相似度（保留原有逻辑）
        total_sim = 0.0
        valid_params = 0
        
        # 计算真实梯度的平均值
        avg_real_grads = {}
        for real_grad in real_grads:
            for name, grad in real_grad.items():
                if name not in avg_real_grads:
                    avg_real_grads[name] = torch.zeros_like(grad)
                avg_real_grads[name] += grad.clone()
        for name in avg_real_grads:
            avg_real_grads[name] = avg_real_grads[name] / len(real_grads)

        # 计算真实梯度的标准差（跨客户端维度）
        real_grad_stds = {}
        for name in avg_real_grads:
            # 收集所有客户端的梯度
            all_grads = [client_grad[name] for client_grad in real_grads]
            # 沿着客户端维度计算标准差
            real_grad_stds[name] = torch.stack(all_grads).std(dim=0)

        # 计算虚假梯度的标准差
        fake_grad_stds = {name: torch.std(grad) for name, grad in fake_grads.items()}

        
        # 遍历所有参数比较梯度
        for param_name in fake_grads:
            if param_name in avg_real_grads:
                # 确保设备一致
                avg_real_grad = avg_real_grads[param_name].to(fake_grads[param_name].device)
                fake_grad = fake_grads[param_name]
                real_std = real_grad_stds.get(param_name, torch.tensor(0.0))
                fake_std = fake_grad_stds.get(param_name, torch.tensor(0.0))
                
                sim = cosine_similarity(avg_real_grad, fake_grad)
                total_sim += sim
                valid_params += 1
                print(f"[{param_name}]")
                print(f"  Cosine Similarity: {sim:.4f}")
                print(f"  Real Grad Norm: {avg_real_grad.norm().item():.4f}")
                
                if real_std.numel() > 1:
                    real_std_print = real_std.mean().item()
                else:
                    real_std_print = real_std.item()
                print(f"  Real Grad Std: {real_std_print:.4f}") 
                print(f"  Fake Grad Norm: {fake_grad.norm().item():.4f}")

                if fake_std.numel() > 1:
                    fake_std_print = fake_std.mean().item()
                else:
                    fake_std_print = fake_std.item()
                print(f"  Fake Grad Std: {fake_std.item():.4f}")    # 新增标准差输出

        # 输出平均相似度
        if valid_params > 0:
            avg_sim = total_sim / valid_params
            print(f"\nAverage Cosine Similarity: {avg_sim:.4f}")
        else:
            print("\nNo matching parameters for comparison")