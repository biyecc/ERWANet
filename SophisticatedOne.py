import torch
import torch.nn as nn
import torch_geometric.nn as pyg
import numpy as np

from heteroconv import HeteroConv
from sophis_conv1 import SAGEConv
from components.TransformerPooling import GraphMultisetTransformer


class SophisticatedModel(torch.nn.Module):
    def __init__(self, num_layers=4, mdim=512, edge_embed=True, global_embed=True):
        super().__init__()
        # edge_channel需要一个合适的表示（这什么量纲啊？）
        hidden_channels = 512
        out_channels = 250
        input_channel = mdim
        self.num_layers = num_layers
        self.edge_embed, self.global_embed = edge_embed, global_embed
        self.pretransform_win = pyg.Linear(input_channel, hidden_channels, bias=False)
        self.post_plot_transform = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            pyg.Linear(hidden_channels, hidden_channels, bias=False),
            nn.LeakyReLU(0.2, True),
        )
        # self.pretransform_edge = pyg.Linear(edge_channel, hidden_channels, bias=False)
        # self.post_edge_transform = nn.Sequential(
        #     nn.LeakyReLU(0.2, True),
        #     pyg.Linear(hidden_channels, hidden_channels, bias=False),
        #     nn.LeakyReLU(0.2, True),
        # )
        self.leaklyrelu = nn.LeakyReLU(0.2)
        #####################
        # 点嵌入更新块儿
        self.plot_convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('window', 'near', 'window'): SAGEConv(hidden_channels, hidden_channels, edge_embed=edge_embed),
                ('window', 'close', 'window'): SAGEConv(hidden_channels, hidden_channels, edge_embed=edge_embed),
                ('window', 'sim', 'window'): SAGEConv(hidden_channels, hidden_channels, edge_embed=edge_embed),
                # ('window', 'sim', 'window'): pyg.SAGEConv((hidden_channels,hidden_channels), hidden_channels, hidden_channels, add_self_loops = False),
            }, aggr='mean')
            self.plot_convs.append(conv)

        # # 边嵌入更新块儿
        # self.edge_convs = nn.ModuleList()
        # # 点嵌入更新块儿
        # for _ in range(num_layers):
        #     conv = HeteroConv({
        #         ('edge', '0', 'edge'): pyg.SAGEConv(hidden_channels, hidden_channels),
        #         ('edge', '1', 'edge'): pyg.SAGEConv(hidden_channels, hidden_channels),
        #         ('edge', '2', 'edge'): pyg.SAGEConv(hidden_channels, hidden_channels),
        #         ('edge', '3', 'edge'): pyg.SAGEConv(hidden_channels, hidden_channels),
        #         ('edge', '4', 'edge'): pyg.SAGEConv(hidden_channels, hidden_channels),
        #         ('edge', '5', 'edge'): pyg.SAGEConv(hidden_channels, hidden_channels),
        #         ('edge', '6', 'edge'): pyg.SAGEConv(hidden_channels, hidden_channels),
        #         ('edge', '7', 'edge'): pyg.SAGEConv(hidden_channels, hidden_channels)
        #     }, aggr='mean')
        #     self.edge_convs.append(conv)
        if global_embed:
            self.poolers = nn.ModuleList()
            for _ in range(num_layers):
                self.poolers.append(GraphMultisetTransformer(hidden_channels, hidden_channels, hidden_channels, None,
                                                             pool_sequences=['GMPool_I']))  # 只取最后一个环节获得一个节点

        # self.pool = CSRA(hidden_channels)
        self.lin = pyg.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, x_pos, edge_index_dict):
        # 数据集要求：点属性，边属性，以及配套的ij2idx来取边属性
        # edge_index, edge_edge_index
        x_dict['window'] = self.post_plot_transform(self.pretransform_win(x_dict['window']))
        batch_idx = torch.Tensor(np.zeros(len(x_dict), dtype="int64"))
        # x_dict['edge'] = self.post_edge_transform(self.pretransform_edge(x_dict['edge']))

        for l in range(self.num_layers):
            global_vec = None
            if self.global_embed:
                global_vec = self.poolers[l](x_dict['window'], batch=None)
            x_dict = self.plot_convs[l](x_dict, x_pos, edge_index_dict, global_vec)  # 这地方要改，引入边的信息
            x_dict = {key: self.leaklyrelu(x) for key, x in x_dict.items()}

        # return self.lin(self.pool(x_dict, edge_index_dict))
        return self.lin(x_dict['window'])
