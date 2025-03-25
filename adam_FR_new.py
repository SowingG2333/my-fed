import copy
import torch
import torch.optim as optim

class AdamFreeRider:
    """生成经Adam优化器处理的虚假梯度，仅影响最后一层"""
    def __init__(self, cid, global_model, lr=0.01, betas=(0.9, 0.999), eps=1e-8, sigma_n=0.1):
        self.cid = cid
        self.global_model = copy.deepcopy(global_model)
        self.local_model = copy.deepcopy(global_model)
        self.sigma_n = sigma_n
        
        # 识别最后一层参数
        param_names = [name for name, _ in self.local_model.named_parameters()]
        last_param_name = param_names[-1]
        self.last_layer_prefix = last_param_name.rsplit('.', 1)[0]
        self.last_layer_params = [name for name in param_names if name.startswith(self.last_layer_prefix)]
        
        # 初始化仅针对最后一层的Adam优化器
        last_layer_parameters = [param for name, param in self.local_model.named_parameters()
                                 if name in self.last_layer_params]
        self.optimizer = optim.Adam(
            last_layer_parameters,
            lr=lr,
            betas=betas,
            eps=eps
        )
        
        self.prev_local_state = None  # 保存上一轮参数状态

    def update_global_model(self, new_global_state):
        """更新本地模型参数（不替换张量对象）"""
        # 保存当前参数用于后续梯度计算
        self.prev_local_state = copy.deepcopy(self.local_model.state_dict())
        
        # 将全局模型参数复制到本地（保留张量引用）
        local_state = self.local_model.state_dict()
        for key in new_global_state:
            if key in local_state:
                local_state[key].data.copy_(new_global_state[key])

    def generate_fake_grad(self, round_num):
        """生成经过Adam优化器处理的虚假梯度"""
        # 保存原始参数状态
        original_state = copy.deepcopy(self.local_model.state_dict())
        fake_gradients = {}

        # 根据轮次生成基础梯度
        if round_num == 0:
            # 首轮生成随机噪声
            for name in self.last_layer_params:
                param = self.local_model.state_dict()[name]
                noise = torch.normal(mean=0, std=self.sigma_n, size=param.size())
                fake_gradients[name] = noise
        else:
            # 后续轮次计算参数差异作为梯度
            for name in self.last_layer_params:
                curr_param = self.local_model.state_dict()[name]
                prev_param = self.prev_local_state[name]
                fake_gradients[name] = curr_param - prev_param.to(curr_param.device)

        # 设置梯度并执行Adam更新
        for name, param in self.local_model.named_parameters():
            param.grad = fake_gradients[name].clone() if name in fake_gradients else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        # 计算Adam实际应用的更新量
        updated_state = self.local_model.state_dict()
        adam_gradients = {
            name: (updated_state[name] - original_state[name]).detach()
            for name in self.last_layer_params
        }

        # 恢复原始参数
        self.local_model.load_state_dict(original_state)
        
        return adam_gradients

def cosine_similarity(grad1, grad2):
    """计算两个梯度的余弦相似度"""
    flat_grad1 = grad1.flatten()
    flat_grad2 = grad2.flatten()
    dot_product = torch.dot(flat_grad1, flat_grad2)
    norm_product = torch.norm(flat_grad1) * torch.norm(flat_grad2)
    return (dot_product / norm_product).item() if norm_product != 0 else 0.0

if __name__ == '__main__':
    # 初始化环境
    from fed_global import FedModel
    from normal_client import NormalClient
    from main import data_generate

    global_model = FedModel()
    normal_client = NormalClient(1, global_model, 0.1, 64, 30, data_generate(1)[0])
    free_rider = AdamFreeRider(
        cid=0,
        global_model=global_model,
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        sigma_n=0.5
    )

    # 模拟三轮联邦学习
    for round_num in range(10):
        print(f"\n=== Round {round_num} ===")
        
        # 获取新全局模型和真实梯度
        new_global_model_state_dict, real_grads = normal_client.local_train()
        free_rider.update_global_model(new_global_model_state_dict)
        
        # 生成虚假梯度
        fake_grads = free_rider.generate_fake_grad(round_num)
        
        # 计算余弦相似度
        total_sim = 0.0
        valid_params = 0
        
        # 遍历所有参数比较梯度
        for param_name in fake_grads:
            if param_name in real_grads:
                # 确保张量在相同设备
                real_g = real_grads[param_name].to(fake_grads[param_name].device)
                sim = cosine_similarity(real_g, fake_grads[param_name])
                total_sim += sim
                valid_params += 1
                print(f"[{param_name}]")
                print(f"  Cosine Similarity: {sim:.4f}")
                print(f"  Real Grad Norm: {real_g.norm().item():.4f}")
                print(f"  Fake Grad Norm: {fake_grads[param_name].norm().item():.4f}")

        # 输出平均相似度
        if valid_params > 0:
            avg_sim = total_sim / valid_params
            print(f"\nAverage Cosine Similarity: {avg_sim:.4f}")
        else:
            print("\nNo matching parameters for comparison")