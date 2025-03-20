from adam_FR import AdamFreeRider                                       # 引入adam优化器搭便车攻击模块
from normal_client import NormalClient                                  # 引入正常客户端模块
from fed_global import FedModel, non_iid_split, split_dataset_randomly  # 引入全局模型架构与非独立同分布数据划分
from cos_defender import CosineDefender                                 # 引入余弦相似度检测器
from torchvision import datasets, transforms                            # 引入torchvision的数据集与变换模块
import numpy as np                                                      # 引入numpy模组

# 数据划分
def data_generate(num_clients, non_iid=True):
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
    if non_iid == True:
        client_datasets = non_iid_split(full_dataset, num_clients)
    else:
        client_datasets = split_dataset_randomly(full_dataset, num_clients)

    return client_datasets

# test
if __name__ == '__main__':
    '''全局参数定义'''
    NUM_CLIENTS = 50                 # 客户端总数
    MALICIOUS_RATIO = 0.25            # 恶意客户端比例
    NUM_ROUNDS = 30                  # 联邦训练轮次
    CHOOSE_PERCENTAGE = 0.5          # 每轮学习参与率
    COS_THRESHOLD = 0.5              # 防御阈值
    LOCAL_EPOCHS = 50                # 本地训练轮次
    BATCH_SIZE = 64                  # 本地训练批次大小
    MODEL_LR = 0.1                   # 模型学习率
    ATTACK_PARAMS = (0.8, 0.1)       # 攻击参数(α, β)

    # 数据集划分与实例化
    non_iid_datasets = data_generate(NUM_CLIENTS)
    global_model = FedModel()
    defender = CosineDefender(NUM_CLIENTS * CHOOSE_PERCENTAGE, COS_THRESHOLD, global_model)

    # 选择恶意客户端
    all_clients = list(range(NUM_CLIENTS))
    fr_clients = np.random.choice(all_clients, 
                                  size=int(NUM_CLIENTS * MALICIOUS_RATIO), 
                                  replace=False).tolist()
    
    # 初始化所有客户端
    all_clients = {}
    for cid in range(NUM_CLIENTS):
        if cid in fr_clients:
            # fr客户端
            all_clients[cid] = {
                'type': 'fr',
                'trainer': AdamFreeRider(cid, global_model, MODEL_LR, ATTACK_PARAMS)
            }
        else:
            # 正常客户端
            all_clients[cid] = {
                'type': 'normal',
                'trainer': NormalClient(cid, 
                                        global_model, 
                                        MODEL_LR, 
                                        BATCH_SIZE, 
                                        LOCAL_EPOCHS,
                                        non_iid_datasets[cid])      # 此处的数据集是按照客户端数划分的，但是fr客户端不会利用真实数据进行训练
            }

    # 联邦学习循环
    for round_num in range(NUM_ROUNDS):
        print(f"round {round_num}")
        # 选择参与客户端
        selected_clients = np.random.choice(
            list(all_clients.keys()), 
            size=int(NUM_CLIENTS * CHOOSE_PERCENTAGE),
            replace=False
        )
        # 收集梯度更新
        update_grads = []
        for cid in selected_clients:
            current_client = all_clients[cid]
            # 如果是恶意客户端
            if current_client['type'] == 'fr':
                grad = current_client['trainer'].generate_fake_grad(round_num)
            else:
                grad = current_client['trainer'].local_train()

            # 记录梯度更新
            update_grads.append(grad)
            defender.add_update(cid, grad)

        # 执行防御检测
        detected_FRs = defender.FR_detection()
        detected_results = {}
        for (cid, cos_sim) in detected_FRs:
            detected_results[cid] = {
                'cos_sim': cos_sim,
                'fr_or_not': 'yes' if cos_sim < COS_THRESHOLD else 'not'
            }

        # 衡量防御性能
        detected_count = 0
        for cid in range(NUM_CLIENTS):
            if (detected_results[cid]['fr_or_not'] == 'yes') and (all_clients[cid]['type'] == 'fr'):
                detected_count += 1
        detected_ratio = detected_count / (NUM_CLIENTS * MALICIOUS_RATIO)
        print(detected_ratio)