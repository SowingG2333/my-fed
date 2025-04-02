import copy
import torch
import torch.optim as optim

class SimpleAdamFreeRider:
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
    def __init__(self, cid, global_model, lr, betas, eps, sigma_n):
        # 初始化cid、全局模型、本地模型、首轮生成的噪声标准差
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