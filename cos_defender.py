import torch                                # 引入torch模块
import torch.functional as F                # 引入torch的函数模块

class CosineDefender:
    """基于余弦相似度计算的搭便车攻击防御器"""
    def __init__(self, max_num, cos_threshold, model):
        self.c_num = 0                       # 当前累积的客户端数量
        self.max_num = max_num               # 触发检测的客户端数量阈值
        self.c_ids = []                      # 储存客户端id（用于检测时的映射）
        self.avg_gradients = {}              # 梯度累积字典
        self.gradients = []                  # 存储所有客户端梯度
        self.cos_threshold = cos_threshold   # 相似度阈值
        self.model = model                   # 参考模型（用于参数结构）

    def FR_detection(self):
        """检测恶意客户端，返回恶意客户端索引列表"""
        # 计算平均梯度向量
        avg_grad = {name: grad / self.c_num for name, grad in self.avg_gradients.items()}
        avg_vector = torch.cat([v.flatten() for v in avg_grad.values()]).detach()

        # 检测低相似度客户端
        malicious = []
        for i, client_grad in enumerate(self.gradients):
            # 展平客户端梯度
            client_vector = torch.cat([v.flatten() for v in client_grad.values()]).detach()
            
            # 计算余弦相似度
            similarity = F.cosine_similarity(avg_vector.unsqueeze(0), 
                                             client_vector.unsqueeze(0), 
                                             dim=1).item()
            
            if similarity < self.cos_threshold:
                malicious.append((self.c_ids[i], similarity))
        
        return malicious    # 返回形式 [(id, 相似度分数)...]

    def add_update(self, client_id, client_grads):
        """添加客户端梯度更新"""
        if self.c_num <= self.max_num:
            self.c_ids.append(client_id)
            # 存储梯度并累加
            self.gradients.append(client_grads)
            for name, grad in client_grads.items():
                if name not in self.avg_gradients:
                    self.avg_gradients[name] = grad.clone()
                else:
                    self.avg_gradients[name] += grad
            self.c_num += 1
            return None
        else:
            # 触发检测并重置
            malicious_clients = self.FR_detection()
            print(f"触发异步聚合，检测到恶意客户端索引: {malicious_clients}")
            self.clear_avg_grad()
            self.c_num = 0
            self.gradients = []
            return malicious_clients  # 返回恶意客户端列表

    def clear_avg_grad(self):
        """清空累积梯度和id"""
        self.avg_gradients = {}
        self.c_ids = []