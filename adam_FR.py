import torch                    # 引入torch模块
import copy                     # 引入深拷贝模块
import torch.optim as optim     # 引入torch的优化器模块

# class AdamFreeRider:
#     """实现联邦学习中对抗性梯度生成的优化器"""
#     def __init__(self, cid, global_model, lr, adam_params):
#         self.cid = cid
#         self.model = copy.deepcopy(global_model)
#         self.optimzier = optim.Adam(self.model.parameters(), lr, adam_params)
#         self.alpha, self.beta = adam_params[0], adam_params[1]
#         self.fake_gradients = {}

#     def generate_fake_grad(self, round_num):
#         """生成根据adam优化器公式的梯度更新"""
#         for name, param in self.model.named_parameters():
#             if round_num == 0:
#                 # 生成首轮噪声梯度
#                 noise = torch.rand_like(param)
#                 param.grad = noise
#                 self.fake_gradients[name] = noise

#             else:
#             # 生成融合梯度
#                 base_grad = self.fake_gradients[name] * self.alpha
#                 noise = torch.randn_like(param) * self.beta
#                 fake_grad = base_grad + noise

#                 # 梯度赋值
#                 param.grad = noise
#                 self.fake_gradients[name] = fake_grad

#         # 执行Adam优化器更新
#         self.optimzier.step()
#         self.optimzier.zero_grad()

#         return self.fake_gradients

# 只返回最后一层的版本
class AdamFreeRider:
    """实现联邦学习中仅针对最后一层参数的对抗性梯度生成的优化器"""
    def __init__(self, cid, global_model, lr, adam_params):
        self.cid = cid
        self.model = copy.deepcopy(global_model)
        
        # 获取所有参数的名称并确定最后一层的前缀
        param_names = [name for name, _ in self.model.named_parameters()]
        last_param_name = param_names[-1]
        self.last_layer_prefix = last_param_name.rsplit('.', 1)[0]
        self.last_layer_params = [name for name in param_names if name.startswith(self.last_layer_prefix)]
        
        # 仅将最后一层参数传递给优化器
        last_layer_parameters = [param for name, param in self.model.named_parameters() 
                                 if name in self.last_layer_params]
        self.optimizer = optim.Adam(last_layer_parameters, lr, adam_params)
        
        self.alpha, self.beta = adam_params[0], adam_params[1]
        self.fake_gradients = {}

    def generate_fake_grad(self, round_num):
        """生成仅针对最后一层参数的伪造梯度更新"""
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

        # 执行优化器更新（仅影响最后一层参数）
        self.optimizer.step()
        self.optimizer.zero_grad()

        # 更新并返回仅包含最后一层的伪造梯度
        self.fake_gradients.update(current_gradients)
        return self.fake_gradients

# test
from fed_global import FedModel

if __name__ == '__main__':
    model = FedModel()
    adam_cli = AdamFreeRider(global_model=model, cid=0, lr=0.01, adam_params=(0.8, 0.1))

    for round_num in range(3):
        print(f"{round_num}")
        fake_gradients = adam_cli.generate_fake_grad(round_num)