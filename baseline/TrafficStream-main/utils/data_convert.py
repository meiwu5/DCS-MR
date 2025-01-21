import numpy as np
import os
import os.path as osp
import pdb
import networkx as nx
from utils.common_tools import mkdirs
import tqdm
import random

def z_score(data):
    std = np.std(data)
    return (data - np.mean(data)) / std if std != 0 else data


def generate_dataset(data, idx, x_len=12, y_len=12):
    res = data[idx]
    node_size = data.shape[1]
    t = len(idx)-1
    idic = 0
    x_index, y_index = [], []
    
    for i in tqdm.tqdm(range(t,0,-1)):
        if i-x_len-y_len>=0:
            x_index.extend(list(range(i-x_len-y_len, i-y_len)))
            y_index.extend(list(range(i-y_len, i)))

    x_index = np.asarray(x_index)
    y_index = np.asarray(y_index)
    x = np.nan_to_num(res[x_index].reshape((-1, x_len, node_size)))
    y = np.nan_to_num(res[y_index].reshape((-1, y_len, node_size)))
 
    return x, y

def generate_samples(days, savepath, data, graph, train_rate=0.6, val_rate=0.2, test_rate=0.2, val_test_mix=False):
    edge_index = np.array(list(graph.edges)).T
    del graph
    data = data[0:days*288, :]
    t, n = data.shape[0], data.shape[1]
    
    train_idx = [i for i in range(int(t*train_rate))]
    val_idx = [i for i in range(int(t*train_rate), int(t*(train_rate+val_rate)))]
    test_idx = [i for i in range(int(t*(train_rate+val_rate)), t)]
    
    train_x, train_y = generate_dataset(data, train_idx)
    val_x, val_y = generate_dataset(data, val_idx)
    test_x, test_y = generate_dataset(data, test_idx)
    if val_test_mix:
        val_test_x = np.concatenate((val_x, test_x), 0)
        val_test_y = np.concatenate((val_y, test_y), 0)
        val_test_idx = np.arange(val_x.shape[0]+test_x.shape[0])
        np.random.shuffle(val_test_idx)
        val_x, val_y = val_test_x[val_test_idx[:int(t*val_rate)]], val_test_y[val_test_idx[:int(t*val_rate)]]
        test_x, test_y = val_test_x[val_test_idx[int(t*val_rate):]], val_test_y[val_test_idx[int(t*val_rate):]]

    train_x = z_score(train_x)
    val_x = z_score(val_x)
    test_x = z_score(test_x)
    np.savez(savepath, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, edge_index=edge_index)
    data = {"train_x":train_x, "train_y":train_y, "val_x":val_x, "val_y":val_y, "test_x":test_x, "test_y":test_y, "edge_index":edge_index}
    return data


# flow_folder = "/root/autodl-fs/TrafficStream-main/data/PEMSD8-stream/finaldata"
# graph_folder = "/root/autodl-fs/TrafficStream-main/data/PEMSD8-stream/graph"
# save_folder = "/root/autodl-fs/TrafficStream-main/data/PEMSD8-stream/FastData"
# mkdirs(save_folder)

# # 定义生成数据集的参数
# days = 30  # 使用的天数
# train_rate = 0.6
# val_rate = 0.2
# test_rate = 0.2
# x_len = 12
# y_len = 12

# # 遍历 flow 文件和 graph 文件
# for year in range(2012, 2018 + 1):
#     flow_file_path = os.path.join(flow_folder, f"{year}.npz")
#     graph_file_path = os.path.join(graph_folder, f"{year}_adj.npz")
#     save_path = os.path.join(save_folder, f"{year}_30day.npz")

#     if os.path.exists(flow_file_path) and os.path.exists(graph_file_path):
#         # 加载 flow 数据
#         flow_data = np.load(flow_file_path)['x']
        
#         # 加载 graph 数据并创建无向图
#         adjacency_matrix = np.load(graph_file_path)['x']
#         graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.Graph)
        
#         # 生成数据集
#         data = generate_samples(days, save_path, flow_data, graph, train_rate, val_rate, test_rate, val_test_mix=False)
#         print(f"Dataset for {year} saved at {save_path}")
#     else:
#         print(f"Data for year {year} not found in specified paths.")


