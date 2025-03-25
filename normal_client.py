import copy                                 # 引入深拷贝模块
import torch.optim as optim                 # 引入torch的优化器模块
import torch.nn as nn                       # 引入torch的神经网络模块
from torch.utils.data import DataLoader     # 引入torch的数据集加载模块

# class NormalClient:
#     """正常客户端训练器"""
#     def __init__(self, cid, global_model, lr, batch_size, local_epochs, dataset):
#         self.cid = cid
#         self.global_model = global_model
#         self.local_model = copy.deepcopy(global_model)
#         self.optimizer = optim.SGD(self.local_model.parameters(), lr=lr)
#         self.criterion = nn.CrossEntropyLoss()
#         self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         self.local_epochs = local_epochs

#     def local_train(self):
#         """执行本地训练并返回梯度更新"""
#         # 保留初始参数（注意使用的是state_dict）
#         initial_params = copy.deepcopy(self.local_model.state_dict())
        
#         # 训练过程
#         self.local_model.train()
#         for _ in range(self.local_epochs):
#             for X, y in self.loader:
#                 self.optimizer.zero_grad()
#                 pred = self.local_model(X)
#                 loss = self.criterion(pred, y)
#                 loss.backward()
#                 self.optimizer.step()
        
#         # 计算梯度更新
#         # 注意此时修改为返回最后一层参数
#         grad_update = {
#             name: initial_params[name] - self.local_model.state_dict()[name] 
#             for name in initial_params.keys()
#         }
#         return grad_update

# 仅返回最后一层参数的版本
class NormalClient:
    """仅返回最后一层参数更新的正常客户端训练器"""
    def __init__(self, cid, global_model, lr, batch_size, local_epochs, dataset):
        self.cid = cid
        self.global_model = global_model
        self.local_model = copy.deepcopy(global_model)
        
        # 识别最后一层参数
        param_names = [name for name, _ in self.local_model.named_parameters()]
        last_param_name = param_names[-1]
        self.last_layer_prefix = last_param_name.rsplit('.', 1)[0]
        self.last_layer_params = [name for name in param_names if name.startswith(self.last_layer_prefix)]
        
        # 配置仅优化最后一层参数
        last_layer_parameters = [param for name, param in self.local_model.named_parameters() 
                                if name in self.last_layer_params]
        self.optimizer = optim.SGD(last_layer_parameters, lr=lr)
        
        self.criterion = nn.CrossEntropyLoss()
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.local_epochs = local_epochs

    def local_train(self):
        """执行本地训练并返回最后一层梯度更新"""
        # 保留初始参数（深拷贝state_dict）
        initial_params = copy.deepcopy(self.local_model.state_dict())
        
        # 训练过程（返回最后一层更新）
        for _ in range(self.local_epochs):
            for X, y in self.loader:
                self.optimizer.zero_grad()
                pred = self.local_model(X)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
        
        # 计算最后一层参数更新
        final_params = self.local_model.state_dict()
        grad_update = {
            name: initial_params[name] - final_params[name]
            for name in self.last_layer_params
        }
        return self.local_model.state_dict(), grad_update