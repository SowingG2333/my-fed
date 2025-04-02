import copy                                 # 引入深拷贝模块
import torch.optim as optim                 # 引入torch的优化器模块
import torch.nn as nn                       # 引入torch的神经网络模块
from torch.utils.data import DataLoader     # 引入torch的数据集加载模块

class NormalClient:
    """仅返回最后一层参数更新的正常客户端训练器"""
    def __init__(self, cid, device, 
                 global_model, lr, optimizer, betas, eps, 
                 batch_size, local_epochs, dataset):
        # 初始化客户端ID、设备、全局模型、本地模型、最后一层参数前缀
        self.cid = cid
        self.device = device
        self.global_model = global_model
        self.local_model = copy.deepcopy(global_model).to(device)
        
        # 识别最后一层参数
        param_names = [name for name, _ in self.local_model.named_parameters()]
        last_param_name = param_names[-1]
        self.last_layer_prefix = last_param_name.rsplit('.', 1)[0]
        self.last_layer_params = [name for name in param_names if name.startswith(self.last_layer_prefix)]
        
        # 配置仅优化最后一层参数
        last_layer_parameters = [param for name, param in self.local_model.named_parameters() 
                                if name in self.last_layer_params]
        
        # 初始化优化器、损失函数、数据加载器、本地训练轮次
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(last_layer_parameters, lr=lr)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.local_model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
            )
        
        # 损失函数和数据加载器
        self.criterion = nn.CrossEntropyLoss()
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.local_epochs = local_epochs

    def update_global_model(self, new_global_state):
        """更新本地模型参数"""
        local_state = self.local_model.state_dict()
        for key in new_global_state:
            if key in local_state:
                # 将目标参数显式移动到本地模型所在的设备，再执行复制
                local_state[key].data.copy_(new_global_state[key].to(self.device))

    def local_train(self):
        """执行本地训练并返回最后一层梯度更新"""
        # 保留初始参数用于计算梯度更新
        initial_params = {
            name: param.clone().detach().to(self.device)
            for name, param in self.local_model.state_dict().items()
        }
        
        # 训练过程
        self.local_model.train()
        for _ in range(self.local_epochs):
            for X, y in self.loader:
                # 将数据迁移到模型所在的设备
                X, y = X.to(self.device), y.to(self.device)
                # 梯度清零
                self.optimizer.zero_grad()
                pred = self.local_model(X)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
        
        # 计算最后一层参数更新
        final_params = {
            name: param.to(self.device)  # 显式指定设备
            for name, param in self.local_model.state_dict().items()
        }
        grad_update = {
            name: initial_params[name] - final_params[name]
            for name in self.last_layer_params
        }
        
        return self.local_model.state_dict(), grad_update