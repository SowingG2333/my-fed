import copy                                 # 引入深拷贝模块
import torch.optim as optim                 # 引入torch的优化器模块
import torch.nn as nn                       # 引入torch的神经网络模块
from torch.utils.data import DataLoader     # 引入torch的数据集加载模块

class NormalClient:
    """正常客户端训练器"""
    def __init__(self, cid, global_model, lr, batch_size, local_epochs, dataset):
        self.cid = cid
        self.global_model = global_model
        self.local_model = copy.deepcopy(global_model)
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.local_epochs = local_epochs

    def local_train(self):
        """执行本地训练并返回梯度更新"""
        # 保留初始参数
        initial_state = copy.deepcopy(self.local_model.state_dict())
        
        # 训练过程
        self.local_model.train()
        for _ in range(self.local_epochs):
            for X, y in self.loader:
                self.optimizer.zero_grad()
                pred = self.local_model(X)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
        
        # 计算梯度更新（当前参数与初始参数的差值）
        grad_update = {
            name: initial_state[name] - self.local_model.state_dict()[name] 
            for name in initial_state.keys()
        }
        return grad_update