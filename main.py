from adam_FR import AdamFreeRider               # 引入adam优化器搭便车攻击模块
from normal_client import NormalClient          # 引入正常客户端模块
from fed_global import FedModel, non_iid_split  # 引入全局模型架构与非独立同分布数据划分
from torchvision import datasets, transforms    # 引入torchvision的数据集与变换模块

def data_generate(num_clients):
    # 使用MNIST数据集并进行预处理归一化
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载完整数据集
    full_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 划分非独立同分布客户端数据集
    client_datasets = non_iid_split(full_dataset, num_clients)

    return client_datasets

if __name__ == 'main':
    '''全局参数定义'''
    NUM_CLIENTS = 20                 # 客户端总数
    MALICIOUS_RATIO = 0.1            # 恶意客户端比例
    NUM_ROUNDS = 30                  # 联邦训练轮次
    CHOOSE_PERCENTAGE = 0.5          # 每轮学习参与率
    COS_THRESHOLD = 0.5              # 防御阈值
    LOCAL_EPOCHS = 10                # 本地训练轮次
    BATCH_SIZE = 64                  # 本地训练批次大小
    MODEL_LR = 0.1                   # 模型学习率
    ATTACK_PARAMS = (0.8, 0.1)       # 攻击参数(α, β)