import sys
sys.path.append('src/')
import numpy as np
from scipy.stats import entropy as kldiv
from utils.dataloader import Cotinual_learning_DataLoader
import torch
from scipy.spatial import distance
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import pickle

def contrastive_loss(embedding1, embedding2, temperature=0.1):
    sim_matrix = F.cosine_similarity(embedding1.unsqueeze(1), embedding2.unsqueeze(0), dim=-1)
    sim_matrix = sim_matrix / temperature
    labels = torch.arange(sim_matrix.size(0)).to(embedding1.device)
    
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def get_feature(data, graph, args, model, adj):
    node_size = data.shape[1]
    data = np.reshape(data[-288*7-1:-1,:], (-1, args.x_len, node_size, 3))
    dataloader = Cotinual_learning_DataLoader(data, batch_size=data.shape[0], shuffle=True,pad_with_last_sample=True)
    for batch_idx, data in enumerate(dataloader.get_iterator()):
        feature = model.target_branch(data,args.year)
        return feature.cpu().detach().numpy()
    

def get_current(data, graph, args, model, adj):
    node_size = data.shape[1]
    data = np.reshape(data[-288*7-1:-1,:], (-1, args.x_len, node_size, 3))
    dataloader = Cotinual_learning_DataLoader(data, batch_size=data.shape[0], shuffle=True,pad_with_last_sample=True)
    for batch_idx, data in enumerate(dataloader.get_iterator()):
        feature = model(data,args.year)
        return feature.cpu().detach().numpy()


def get_adj(year, args):
    adj = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]
    adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
    return torch.from_numpy(adj).to(torch.float).to(args.device)
    

def score_func(pre_data, cur_data, args):
    node_size = pre_data.shape[1]
    score = []
    for node in range(node_size):
        max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))
        min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
        pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))
        pre_prob = pre_prob *1.0 / sum(pre_prob)
        cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
        cur_prob = cur_prob * 1.0 /sum(cur_prob)
        score.append(kldiv(pre_prob, cur_prob))
    return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]


def influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph):
    save_dis = {}
    if args.replay_strategy == 'original':
        pre_data = pre_data[-288*7-1:-1,:]
        cur_data = cur_data[-288*7-1:-1,:]
        node_size = pre_data.shape[1]
        score = []
        for node in range(node_size):
            max_val = max(np.max(pre_data[:,node,:]), np.max(cur_data[:,node,:]))
            min_val = min(np.min(pre_data[:,node,:]), np.min(cur_data[:,node,:]))
            pre_prob, _ = np.histogram(pre_data[:,node,:], bins=10, range=(min_val, max_val))
            pre_prob = pre_prob *1.0 / sum(pre_prob)
            cur_prob, _ = np.histogram(cur_data[:,node,:], bins=10, range=(min_val, max_val))
            cur_prob = cur_prob * 1.0 /sum(cur_prob)
            score.append(kldiv(pre_prob, cur_prob))
        return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]
    
    elif args.replay_strategy == 'feature':
        model.eval()
        pre_adj = get_adj(args.year-1, args)
        cur_adj = get_adj(args.year, args)
        
        pre_data = get_feature(pre_data, pre_graph, args, model, pre_adj)
        cur_data = get_current(cur_data, cur_graph, args, model, cur_adj)

        score = []
        
        for i in range(pre_data.shape[0]):
            score_ = 0.0
            for j in range(pre_data.shape[2]):
                pre_data[i, :, j] = (pre_data[i, :, j] - np.min(pre_data[i, :, j])) / (np.max(pre_data[i, :, j]) - np.min(pre_data[i, :, j]))
                cur_data[i, :, j] = (cur_data[i, :, j] - np.min(cur_data[i, :, j])) / (np.max(cur_data[i, :, j]) - np.min(cur_data[i, :, j]))
                pre_prob, _ = np.histogram(pre_data[i, :, j], bins=10, range=(0, 1), density=True)
                cur_prob, _ = np.histogram(cur_data[i, :, j], bins=10, range=(0, 1), density=True)
                
                save_dis[j] = [pre_prob,cur_prob]
                score_ += wasserstein_distance(pre_prob, cur_prob)
            
            score.append(score_)
        with open('save_dis.pkl', 'wb') as f:
            pickle.dump(save_dis, f)
        return np.argsort(score)[-args.topk:]
