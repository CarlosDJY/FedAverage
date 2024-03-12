import os
import copy
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, client
from model.WideResNet import WideResNet
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
# 客户端的数量
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
# 随机挑选的客户端的数量
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
# 训练次数(客户端更新次数)
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
# batchsize大小
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
# 模型名称
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
# 学习率
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-dataset',"--dataset",type=str,default="mnist",help="需要训练的数据集")
# 模型验证频率（通信频率）
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
#n um_comm 表示通信次数，此处设置为1k
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def add_backdoor_pattern(img):
    # 对每个图像添加后门特征
    backdoored_image = img.clone()
    backdoored_image[-6:] = 255
    return backdoored_image

def update_client_model(client_name, client_obj, args, net, loss_func, global_params):
    # 为每个客户端创建网络的独立副本，并移动到正确的设备
    client_net = copy.deepcopy(net).to(dev)  # 使用 dev 替代 args['device']
    client_obj.init_optimizer(client_net, args['learning_rate'])
    
    # 执行本地更新
    return client_name, client_obj.localUpdate(args['epoch'], args['batchsize'], client_net, loss_func, client_obj.optimizer, global_params)

def select_gpu():
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        # 获取可用的 GPU 设备数量
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            # 如果有多个 GPU，仅使用 GPU1
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        else:
            # 如果只有一个 GPU，使用 GPU0
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        # 如果 CUDA 不可用，则使用 CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__


    #-----------------------文件保存-----------------------#
    # 创建结果文件夹
    #test_mkdir("./result")
    # path = os.getcwd()
    # 结果存放test_accuracy中
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 新建文件名
    filename = f"result/test_accuracy_{timestamp}.txt"

    # 打开文件并写入测试结果
    test_txt = open(filename, mode="w")
    # 写入测试结果
    test_txt.write("Your test results here...\n")
    #global_parameters_txt = open("global_parameters.txt",mode="a",encoding="utf-8")
    #----------------------------------------------------#
    # 创建最后的结果
    test_mkdir(args['save_path'])

    # 选择 GPU
    select_gpu()
    # 检查当前可见的 GPU 设备
    print("Visible CUDA devices:", os.environ['CUDA_VISIBLE_DEVICES'])
    
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    # 初始化模型
    # mnist_2nn
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    # mnist_cnn
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
    # ResNet网络
    elif args['model_name'] == 'wideResNet':
        net = WideResNet(depth=28, num_classes=10).to(dev)

    ## 如果有多个GPU
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     net = torch.nn.DataParallel(net)

    # 将Tenor 张量 放在 GPU上
    net = net.to(dev)

    '''
        回头直接放在模型内部
    '''
    # 定义损失函数
    loss_func = F.cross_entropy
    # 优化算法的，随机梯度下降法
    # 使用Adam下降法
    opti = optim.Adam(net.parameters(), lr=args['learning_rate'])

    ## 创建Clients群
    '''
        创建Clients群100个
        
        得到Mnist数据
        
        一共有60000个样本
        100个客户端
        IID：
            我们首先将数据集打乱，然后为每个Client分配600个样本。
        Non-IID：
            我们首先根据数据标签将数据集排序(即MNIST中的数字大小)，
            然后将其划分为200组大小为300的数据切片，然后分给每个Client两个切片。
            注： 我觉得着并不是真正意义上的Non—IID
    '''
    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
    for i in [10]:  # 假设要添加4个客户端
        client_name = f'client{i}'  # 定义客户端名称
        myClients.clients_set[client_name].corrupt_client_data(target_label=7, corruption_ratio=0.4)
    
    testDataLoader = myClients.test_data_loader

    # ---------------------------------------以上准备工作已经完成------------------------------------------#
    # 每次随机选取10个Clients
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    # 得到全局的参数
    global_parameters = {}
    # net.state_dict()  # 获取模型参数以共享

    # 得到每一层中全连接层中的名称fc1.weight
    # 以及权重weights(tenor)
    # 得到网络每一层上
    for key, var in net.state_dict().items():
        # print("key:"+str(key)+",var:"+str(var))
        print("张量的维度:"+str(var.shape))
        print("张量的Size"+str(var.size()))
        global_parameters[key] = var.clone()


    # num_comm 表示通信次数，此处设置为1k
    # 通讯次数一共1000次
    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))
        test_txt.write("communicate round {}".format(i+1))
        
        # 对随机选的将100个客户端进行随机排序
        order = np.random.permutation(args['num_of_clients'])
        # 生成个客户端
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        print("Clients: "+str(clients_in_comm))
        test_txt.write("Clients: "+str(clients_in_comm))
        print(type(clients_in_comm)) # <class 'list'>


        sum_parameters = None
        # 每个Client基于当前模型参数和自己的数据训练并更新模型
        # 返回每个Client更新后的参数
        '''
            import time
            import tqdm
            # 方法1
            # tqdm(list)方法可以传入任意list，如数组
            for i in tqdm.tqdm(range(100)):
               time.sleep(0.5)
               pass
            # 或 string的数组
            for char in tqdm.tqdm(['a','n','c','d']):
               time.sleep(0.5)
               pass
        '''
        # # 这里的clients_
        # for client in tqdm(clients_in_comm):
        #     # 获取当前Client训练得到的参数
        #     # 这一行代码表示Client端的训练函数，我们详细展开：
        #     # local_parameters 得到客户端的局部变量
        #     local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
        #                                                                  loss_func, opti, global_parameters)
        #     # 对所有的Client返回的参数累加（最后取平均值）
        #     if sum_parameters is None:
        #         sum_parameters = {}
        #         for key, var in local_parameters.items():
        #             sum_parameters[key] = var.clone()
        #     else:
        #         for var in sum_parameters:
        #             sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        
        # 在主训练循环中
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(update_client_model, client_name, myClients.clients_set[client_name], args, copy.deepcopy(net).to(dev), loss_func, global_parameters) for client_name in clients_in_comm]
            for future in as_completed(futures):
                client_name, local_parameters = future.result()
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in local_parameters.items():
                        sum_parameters[key] = var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        
        # 取平均值，得到本次通信中Server得到的更新后的模型参数
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        #test_txt.write("communicate round " + str(i + 1) + str('accuracy: {}'.format(sum_accu / num)) + "\n")

        '''
            训练结束之后，我们要通过测试集来验证方法的泛化性，
            注意:虽然训练时，Server没有得到过任何一条数据，但是联邦学习最终的目的
            还是要在Server端学习到一个鲁棒的模型，所以在做测试的时候，是在Server端进行的
        '''
        #with torch.no_grad():
        # 通讯的频率
        #if (i + 1) % args['val_freq'] == 0:
        #  加载Server在最后得到的模型参数
        #  加载Server在最后得到的模型参数
        net.load_state_dict(global_parameters, strict=True)

        # 测试在干净数据上的表现
        sum_accu = 0
        num = 0
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1
        print("\n"+'Accuracy on clean data: {}'.format(sum_accu / num))
        test_txt.write("Communicate round "+str(i+1)+" Clean data accuracy: "+str(float(sum_accu / num))+"\n")

        # 测试在包含后门特征的数据上的表现
        sum_accu_backdoor = 0
        num = 0
        for data_backdoor, _ in testDataLoader:  # 忽略原始标签
            data_backdoor = data_backdoor.to(dev)
            backdoored_data = torch.stack([add_backdoor_pattern(img.cpu()) for img in data_backdoor]).to(dev)
            backdoored_preds = net(backdoored_data)
            backdoored_preds = torch.argmax(backdoored_preds, dim=1)
            sum_accu_backdoor += (backdoored_preds == 7).float().mean()  # 假设target_label是你想要模型将后门数据错误分类到的类别
            num += 1
        print('Accuracy on backdoored data: {}'.format(sum_accu_backdoor / num))
        test_txt.write("Communicate round "+str(i+1)+" Backdoored data accuracy: "+str(float(sum_accu_backdoor / num))+"\n")

        # 随机选择一个测试样例
        sample_idx = np.random.randint(0, len(testDataLoader.dataset))
        sample_data, sample_label = testDataLoader.dataset[sample_idx]
        sample_data_unsqueezed = sample_data.unsqueeze(0).to(dev)  # 添加batch维度并转移到设备

        # 在未添加后门的样本上进行预测
        net.eval()  # 确保网络处于评估模式
        with torch.no_grad():
            pred_clean = net(sample_data_unsqueezed)
            pred_clean_label = torch.argmax(pred_clean, dim=1)

        # 添加后门特征并在篡改的样本上进行预测
        backdoored_sample_data = add_backdoor_pattern(sample_data.cpu()).to(dev)
        backdoored_sample_data_unsqueezed = backdoored_sample_data.unsqueeze(0)  # 添加batch维度
        with torch.no_grad():
            pred_backdoor = net(backdoored_sample_data_unsqueezed)
            pred_backdoor_label = torch.argmax(pred_backdoor, dim=1)

        # 打印并可视化结果
        print(f"Original Label: {sample_label}")
        print(f"Predicted Label (clean): {pred_clean_label.item()}")
        print(f"Predicted Label (backdoored): {pred_backdoor_label.item()}")
        
        test_txt.write(f"Original Label: {sample_label}")
        test_txt.write(f"Predicted Label (clean): {pred_clean_label.item()}")
        test_txt.write(f"Predicted Label (backdoored): {pred_backdoor_label.item()}")

        #test_txt.close()

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))

        # if (i + 1) % 5 == 0:  # Adjust every 5 epochs
        #     for client_name, client_obj in myClients.clients_set.items():
        #         client_obj.adjust_batchsize(args['batchsize'])
    
    test_txt.close()