import copy
import torch
import torch.optim as optim

class AdamFreeRider:
    """模拟联邦学习中基于Adam优化器 每轮全局模型更新梯度并返回最后一层参数的搭便车节点"""
    def __init__(self, cid, device, public_dataset, global_model, criterion, lr, batch_size, betas, eps, sigma_n):
        # 初始化cid、全局模型、本地模型、首轮生成的噪声标准差
        self.cid = cid
        self.device = device
        self.public_dataset = public_dataset
        self.global_model = copy.deepcopy(global_model)
        self.local_model = copy.deepcopy(global_model)
        self.criterion = criterion
        self.batch_size = batch_size
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
                # noise = torch.normal(mean=0, std=self.sigma_n, size=param.size())
                # fake_grad = noise

                # 使用公共数据集训练本地模型
                self.local_model.train()
                for data, target in self.public_dataset:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.local_model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                # 计算当前参数与上轮参数的差异
                prev_param = self.prev_local_state[name].to(param.device)
                fake_grad = param.data - prev_param

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