import sys, json, argparse, random, re, os, shutil
sys.path.append("src/")
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import math
import os.path as osp
import networkx as nx
import pdb

from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import optim
import torch.multiprocessing as mp
# from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch, k_hop_subgraph

from utils import common_tools as ct
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np, masked_mae
from utils.data_convert import generate_samples
from utils.dataloader import DataLoader
from src.model.model import Basic_Model
from src.model import replay
from utils.feature_data import add_time_features
from sklearn.metrics import mean_absolute_error
import torch.nn.functional as F
result = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}}}
pin_memory = True
n_work = 16 

def update(src, tmp):
    for key in tmp:
        if key!= "gpuid":
            src[key] = tmp[key]

def byol_loss(p, z):
    p = nn.functional.normalize(p, dim=1)
    z = nn.functional.normalize(z, dim=1)
    return (2 - 2 * (p * z).sum(dim=1)).sum()

def load_best_model(args):
    if (args.load_first_year and args.year <= args.begin_year+1) or args.train == 0:
        load_path = args.first_year_model_path
        loss = load_path.split("/")[-1].replace(".pkl", "")
    else:
        loss = []
        for filename in os.listdir(osp.join(args.model_path, args.logname+args.time, str(args.year-1))): 
            loss.append(filename[0:-4])
        loss = sorted(loss)
        load_path = osp.join(args.model_path, args.logname+args.time, str(args.year-1), loss[0]+".pkl")
        
    args.logger.info("[*] load from {}".format(load_path))
    checkpoint = torch.load(load_path, map_location=args.device)
    state_dict = checkpoint["model_state_dict"]
    hidden_states_per_year = checkpoint["hidden_states_per_year"]
    if 'tcn2.weight' in state_dict:
        del state_dict['tcn2.weight']
        del state_dict['tcn2.bias']
    model = Basic_Model(args)
    model.load_state_dict(state_dict)
    model.hidden_states_per_year = hidden_states_per_year
    model = model.to(args.device)
    for year, state in model.hidden_states_per_year.items():
        model.hidden_states_per_year[year] = state.to(model.device)
    save_path = "hidden_states_per_year.pth"
    torch.save(model.hidden_states_per_year, save_path)
    print(f"Hidden states per year saved to {save_path}")
    return model, loss[0]

def init(args):    
    conf_path = osp.join(args.conf)
    info = ct.load_json_file(conf_path)
    info["time"] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    update(vars(args), info)
    vars(args)["path"] = osp.join(args.model_path, args.logname+args.time)
    ct.mkdirs(args.path)
    del info


def init_log(args):
    log_dir, log_filename = args.path, args.logname
    logger = logging.getLogger(__name__)
    ct.mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(log_dir, log_filename+".log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join(log_dir, log_filename+".log"))
    vars(args)["logger"] = logger
    return logger


def seed_set(seed=0):
    max_seed = (1 << 32) - 1
    random.seed(seed)
    np.random.seed(random.randint(0, max_seed))
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


def train(inputs, args):
    # Model Setting
    global result
    path = osp.join(args.path, str(args.year))
    ct.mkdirs(path)

    # Dataset Definition
    if args.strategy == 'incremental' and args.year > args.begin_year:
        train_loader = DataLoader(inputs['train_x'][:, :, args.subgraph.numpy()],inputs['train_y'][:, :, args.subgraph.numpy()], batch_size=args.batch_size, shuffle=True,pad_with_last_sample=True )
        val_loader = DataLoader(inputs['val_x'],inputs['val_y'], batch_size=args.batch_size, shuffle=False,pad_with_last_sample=True) 
        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
    else:
        train_loader = DataLoader(inputs["train_x"],inputs["train_y"], batch_size=args.batch_size, shuffle=True,pad_with_last_sample=True)
        val_loader = DataLoader(inputs["val_x"],inputs["val_y"], batch_size=args.batch_size, shuffle=False,pad_with_last_sample=True)
        vars(args)["sub_adj"] = vars(args)["adj"]
    test_loader = DataLoader(inputs["test_x"],inputs["test_y"], batch_size=args.batch_size, shuffle=False,pad_with_last_sample=True)

    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    # Model Definition
    if args.init == True and args.year > args.begin_year:
        gnn_model, _ = load_best_model(args) 
        model = gnn_model
    else:
        gnn_model = Basic_Model(args).to(args.device)
        model = gnn_model

    # Model Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    global_train_steps = len(train_loader) // args.batch_size +1

    iters = len(train_loader)
    lowest_validation_loss = 1e7
    counter = 0
    patience = 50
    use_time = []
    min_train_loss = float('inf')
    min_val_loss = float('inf')
    for epoch in range(args.epoch):
        training_loss = 0.0
        start_time = datetime.now()

        # Train Model
        model.train()
        cn = 0
        for batch_idx, data in enumerate(train_loader.get_iterator()):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("x shape {}".format(data['x'].shape))
                args.logger.info("sub_adj shape {}".format(args.sub_adj.shape))
            optimizer.zero_grad()
            pred = model(data,args.year)

            if args.strategy == "incremental" and args.year > args.begin_year:
                pred = pred[:,:, args.mapping, :]
                data['y'] = data['y'][:,:, args.mapping, :]
            loss = masked_mae(data['y'], pred, 0)
                
            # loss = byol_loss(pred,feat)
            training_loss += float(loss)                                                                                                                                                        
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            model.update_target_network()
            
            cn += 1

        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss/cn
        min_train_loss = min(min_train_loss, training_loss)
 
        # Validate Model
        model.eval()
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader.get_iterator()):
                pred = model(data,args.year)
                loss = masked_mae_np(data['y'].cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss/cn)
        min_val_loss = min(min_val_loss, validation_loss)
        

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")

        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save({'model_state_dict': gnn_model.state_dict(),"hidden_states_per_year": gnn_model.hidden_states_per_year}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break
    args.logger.info(f"Min training loss: {min_train_loss:.4f}, Min validation: {min_val_loss:.4f}")

    best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")
    best_model = Basic_Model(args)
    checkpoint = torch.load(best_model_path, map_location=args.device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_model.hidden_states_per_year = checkpoint["hidden_states_per_year"]
    best_model = best_model.to(args.device)
    
    # Test Model
    test_model(best_model, args, test_loader)
    result[args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


def test_model(model, args, testset):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for batch_idx, data in enumerate(testset.get_iterator()):
            pred = model(data,args.year)
            loss += masked_mae_np(data['y'].cpu().data.numpy(), pred.cpu().data.numpy(), 0)
            pred_.append(pred.cpu().data.numpy())
            truth_.append(data['y'].cpu().data.numpy())
            cn += 1
        loss = loss/cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        mae = metric(truth_, pred_, args)
        return loss


def metric(ground_truth, prediction, args):
    global result
    pred_time = [3,6,12]
    args.logger.info("[*] year {}, testing".format(args.year))
    for i in pred_time:
        mae = masked_mae_np(ground_truth[:, :i, :,:], prediction[:, :i, :,:], 0)
        rmse = masked_mse_np(ground_truth[:, :i, :,:], prediction[:, :i, :,:], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :i, :,:], prediction[:, :i, :,:], 0)
        args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
        result[i]["mae"][args.year] = mae
        result[i]["mape"][args.year] = mape
        result[i]["rmse"][args.year] = rmse
    return mae


def main(args):
    logger = init_log(args)
    logger.info("params : %s", vars(args))
    ct.mkdirs(args.save_data_path)

    for year in range(args.begin_year, args.end_year+1):
        graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year) + "_adj.npz"))["x"])
        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year
        if args.auto_lr:
            new_lr = args.lr / 2
            vars(args)["lr"] = max(new_lr, 0.001)  # Ensure lr is at least 0.001
        else:
            vars(args)["lr"] = args.lr
        inputs = generate_samples(30, osp.join(args.save_data_path, str(year)+'_30day'), np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"], graph, val_test_mix=True) \
            if args.data_process else np.load(osp.join(args.save_data_path, str(year)+"_30day.npz"), allow_pickle=True)

        args.logger.info("[*] Year {} load from {}_30day.npz".format(args.year, osp.join(args.save_data_path, str(year)))) 
        args.logger.info("lr:{}".format(args.lr))


        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)

        if year == args.begin_year and args.load_first_year:
            model, _ = load_best_model(args)
            test_loader = DataLoader(inputs['test_x'],inputs['test_y'], batch_size=args.batch_size, shuffle=False,pad_with_last_sample=True)
            test_model(model, args, test_loader, pin_memory=True)
            continue

        if year > args.begin_year and args.strategy == "incremental":
            model, _ = load_best_model(args)
            
            node_list = list()
            if args.increase:
                cur_node_size = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"].shape[0]
                pre_node_size = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"].shape[0]
                node_list.extend(list(range(pre_node_size, cur_node_size)))

            if args.replay:
                args.logger.info("[*] replay strategy {}".format(args.replay_strategy))
                pre_data = add_time_features(np.load(osp.join(args.raw_data_path, str(year-1)+".npz"))["x"], add_time_in_day=True, add_day_in_week=True)
                cur_data = add_time_features(np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"],add_time_in_day=True, add_day_in_week=True)
                pre_graph = np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"]).edges)).T
                cur_graph = np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T
                vars(args)["topk"] = int(args.replay_ratio*args.graph_size) 
                influence_node_list = replay.influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph)
                node_list.extend(list(influence_node_list))
            
            node_list = list(set(node_list))
            
            cur_graph = torch.LongTensor(np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T)
            edge_list = list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)
            graph_node_from_edge = set()
            for (u,v) in edge_list:
                graph_node_from_edge.add(u)
                graph_node_from_edge.add(v)
            node_list = list(set(node_list) & graph_node_from_edge)
                
           
            if len(node_list) != 0 :
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_list, num_hops=args.num_hops, edge_index=cur_graph, relabel_nodes=True)
                vars(args)["subgraph"] = subgraph
                vars(args)["subgraph_edge_index"] = subgraph_edge_index
                vars(args)["mapping"] = mapping
            logger.info("number of increase nodes:{}, nodes after {} hop:{}, total nodes this year {}".format\
                        (len(node_list), args.num_hops, args.subgraph.size(), args.graph_size))
            vars(args)["node_list"] = np.asarray(node_list)

        if args.strategy != "retrained" and year > args.begin_year and len(args.node_list) == 0:
            model, loss = load_best_model(args)
            ct.mkdirs(osp.join(args.model_path, args.logname+args.time, str(args.year)))
            torch.save({'model_state_dict': model.state_dict(),"hidden_states_per_year": model.hidden_states_per_year}, osp.join(args.model_path, args.logname+args.time, str(args.year), loss+".pkl"))
            test_loader = DataLoader(inputs['test_x'],inputs['test_y'], batch_size=args.batch_size, shuffle=False,pad_with_last_sample=True)
            test_model(model, args, test_loader, pin_memory=True)
            logger.warning("[*] No increasing nodes at year " + str(args.year) + ", store model of the last year.")
            continue
        

        if args.train:
                train(inputs, args=args)
        else:
            if args.auto_test:
                model, _ = load_best_model(args)
                model.eval()
                test_loader = DataLoader(inputs['test_x'],inputs['test_y'], batch_size=args.batch_size, shuffle=False,pad_with_last_sample=True)
                test_model(model, args, test_loader)

    for i in [3, 6, 12]:
        for j in ['mae', 'rmse', 'mape']:
            info = ""
            total = 0.0
            count = 0
            for year in range(args.begin_year, args.end_year + 1):
                if i in result:
                    if j in result[i]:
                        if year in result[i][j]:
                            value = result[i][j][year]
                            info += "{:.2f}\t".format(value)
                            total += value
                            count += 1
            if count > 0:
                avg = total / count
                logger.info("{}\t{}\t{}\t".format(i, j, info) + "average: {:.2f}".format(avg))
            else:
                logger.info("{}\t{}\t{}\t".format(i, j, info) + "average: N/A")
                
    for year in range(args.begin_year, args.end_year+1):
        if year in result:
            info = "year\t{}\ttotal_time\t{}\taverage_time\t{}\tepoch\t{}".format(year, result[year]["total_time"], result[year]["average_time"], result[year]['epoch_num'])
            logger.info(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", type = str, default = "PEMSD8-stream")
    parser.add_argument("--conf", type = str, default = 'test.json')
    parser.add_argument("--paral", type = int, default = 0)
    parser.add_argument("--gpuid", type = int, default = 1)
    parser.add_argument("--logname", type = str, default = "info")
    parser.add_argument("--load_first_year", type = int, default = 0, help="0: training first year, 1: load from model path of first year")
    # parser.add_argument("--first_year_model_path", type = str, default = "res/PEMSD3-stream/model2025-01-04-15:46:37.216135/2011/13.49.pkl", help='specify a pretrained model root')
    # parser.add_argument("--first_year_model_path", type = str, default = "res/PEMSD4-stream/model2025-01-04-21:10:03.106334/2009/21.96.pkl", help='specify a pretrained model root')
    parser.add_argument("--first_year_model_path", type = str, default = "res/PEMSD8-stream/model2025-01-04-22:13:46.722662/2012/14.4783.pkl", help='specify a pretrained model root')
    args = parser.parse_args()
    init(args)
    seed_set(0)

    device = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["device"] = "cuda:0"
    main(args)