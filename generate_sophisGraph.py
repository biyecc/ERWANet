import os
import sys
from collections import namedtuple
from tqdm import tqdm

import numpy as np
import torch
import torch_geometric
import dgl

sys.path.insert(0, "../")
import argparse

from v1.dataset import TxPDataset
from v1.main import KFOLD

parser = argparse.ArgumentParser()
parser.add_argument("--savename", default="big_neightbors_graphs", type=str) ##############
parser.add_argument("--size", default=256, type=int)
parser.add_argument("--emb_path", default="features", type=str) ##############
parser.add_argument("--data", default="/data/CC/EGN-main/10xgenomics", type=str) ##############

args = parser.parse_args()

def get_split(idx, *ss):
    for i in range(len(ss)-1):
        if idx>=ss[i] and idx<ss[i+1]:
            return i
    return len(ss)-1

def get_edge(x, percent=0.01):
    x_ = torch.tensor(x)
    adjs = []

    for each in tqdm(x):
        adj = torch.norm(each - x, dim=2, p=2)
        adjs.append(adj.squeeze())
    adjs = torch.vstack(adjs)
    threshold = torch.quantile(adjs[:, adjs.shape[1]//2], percent)
    adjs = adjs<threshold
    # 方法一：该方法较为准确
    edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adjs)

    return edge_index

for fold in [0, 1, 2]:
    print(f"______开始处理fold{fold}______")
    savename = args.savename + "/" + str(fold)
    os.makedirs(savename, exist_ok=True)

    temp_arg = namedtuple("arg", ["size", "emb_path", "data"])
    temp_arg = temp_arg(args.size, args.emb_path, args.data)
    train_dataset = TxPDataset(KFOLD[fold][0], None, None, temp_arg, train=True)

    temp_arg = namedtuple("arg", ["size", "emb_path", "data"])
    temp_arg = temp_arg(args.size, args.emb_path, args.data)
    foldername = f"{savename}"
    os.makedirs(foldername, exist_ok=True)

    for iid in range(len(KFOLD[fold][0]) + len(KFOLD[fold][1])):
        dataset = TxPDataset([iid], None, None, temp_arg, train=False)

        dataset.min = train_dataset.min.clone()
        dataset.max = train_dataset.max.clone()

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1
        )
        img_data = []
        for x in loader:
            pos, p, py = x["pos"], x["p_feature"], x["count"]
            img_data.append([pos, p, py])

        data = torch_geometric.data.HeteroData()
        # 点属性相关
        data["window"].pos = torch.cat(([i[0] for i in img_data])).clone()
        data["window"].x = torch.cat(([i[1] for i in img_data])).clone()
        data["window"].x = data["window"].x.squeeze()
        data["window"].y = torch.cat(([i[2] for i in img_data])).clone()

        assert len(data["window"]["pos"]) == len(data["window"]["x"]) == len(data["window"]["y"])

        percent=0.00003
        window_edge = get_edge(torch.cat(([i[1] for i in img_data])).clone(), percent=percent)
        print(f"{percent}时，获得的index shape为：", window_edge.shape)
        data['window', 'near', 'window'].edge_index = window_edge
        pos_edge_index = torch_geometric.nn.knn_graph(data["window"]["pos"], k=5, loop=False)
        data["window", "close", "window"].edge_index = pos_edge_index
        edge_index = torch_geometric.nn.knn_graph(data["window"]["x"], k=5, loop=False)
        data["window", "sim", "window"].edge_index = edge_index

        all_edges = torch.concat([window_edge, pos_edge_index, edge_index], dim=-1)
        data_dgl = dgl.graph((all_edges[0], all_edges[1]))

        # 边属性、ij2idx、和edge的边
        # all_edges = torch.concatenate([window_edge, pos_edge_index, edge_index], dim=-1)
        # edge_edges = get_edge_edge(all_edges)
        # we, pei, ei = len(window_edge), len(pos_edge_index), len(edge_index)
        # ij2idx = {}
        # edge_features = []
        # for i, edge in enumerate(all_edges):
        #     ij2idx[tuple(edge)] = i
        #     r_one_hot_i = get_split(i, [0, we, we+pei, we+pei+ei])
        #     r_one_hot = torch.zeros(3)
        #     r_one_hot[r_one_hot_i] = 1
        #     edge_features.append(
        #         torch.concatenate([data["window"].x[edge[0]], data["window"].x[edge[1]], r_one_hot, edge[0]-edge[1],
        #                       torch.norm(data["window"].pos[edge[0]]-data["window"].pos[edge[1]], p=2)])
        #     )
        # edge_features = torch.concatenate(edge_features)
        # data['edge'].x = edge_features
        # data['edge', 'edge_edge', 'edge'].edge_index = edge_edges
        # data['edge'].ij2idx = ij2idx
        torch.save(data, f"{foldername}/{iid}.pt")
        torch.save(data_dgl, f"{foldername}/dgl_{iid}.pt")



