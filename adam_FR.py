import torch                    # 引入torch模块
import copy                     # 引入深拷贝模块
import torch.optim as optim     # 引入torch的优化器模块

class AdamFreeRider:
    """实现联邦学习中对抗性梯度生成的优化器"""
    def __init__(self, cid, global_model, lr, adam_params):
        self.cid = cid
        self.model = copy.deepcopy(global_model)
        self.optimzier = optim.Adam(self.model.parameters(), lr, adam_params)
        self.alpha, self.beta = adam_params[0], adam_params[1]
        self.fake_gradients = {}

    def generate_fake_grad(self, round_num):
        """生成根据adam优化器公式的梯度更新"""
        for name, param in self.model.named_parameters():
            if round_num == 0:
                # 生成首轮噪声梯度
                noise = torch.rand_like(param)
                param.grad = noise
                self.fake_gradients[name] = noise

            else:
            # 生成融合梯度
                base_grad = self.fake_gradients[name] * self.alpha
                noise = torch.randn_like(param) * self.beta
                fake_grad = base_grad + noise

                # 梯度赋值
                param.grad = noise
                self.fake_gradients[name] = fake_grad

        # 执行Adam优化器更新
        self.optimzier.step()
        self.optimzier.zero_grad()

        return self.fake_gradients
    
# test
from fed_global import FedModel

if __name__ == '__main__':
    model = FedModel()
    adam_cli = AdamFreeRider(global_model=model, cid=0, lr=0.01, adam_params=(0.8, 0.1))

    for round_num in range(3):
        print(f"{round_num}")
        fake_gradients = adam_cli.generate_fake_grad(round_num)