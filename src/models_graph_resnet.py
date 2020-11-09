import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import ChebConv  # noqa


class Kipfblock(torch.nn.Module):
    def __init__(self, n_input, n_hidden=64, K=8, p=0.5, bn=False):
        super(Kipfblock, self).__init__()
        self.conv1 = ChebConv(n_input, n_hidden, K=K)
        self.p = p
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.do_bn = bn
        if bn:
            self.bn = torch.nn.BatchNorm1d(n_hidden)

    def forward(self, x, edge_index):
        if self.do_bn:
            x = F.relu(self.bn(self.conv1(x, edge_index)))
        else:
            x = F.relu(self.conv1(x, edge_index))

        x = F.dropout(x, training=self.training, p=self.p)

        return x


class KipfNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=64, K=8, K_mix=2,
                 cached=True, inout_skipconn=False):
        super(KipfNet, self).__init__()
        self.inout_skipconn = inout_skipconn
        self.Kipfblock1 = Kipfblock(n_input=num_features, n_hidden=nh1, K=K)

        if inout_skipconn:
            self.conv_mix = ChebConv(nh1 + num_features, num_classes, K=K_mix)
        else:
            self.conv_mix = ChebConv(nh1, num_classes, K=K_mix)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.Kipfblock1(x, edge_index)

        if self.inout_skipconn:
            x = torch.cat((x, data.x), 1)
            x = self.conv_mix(x, edge_index)
        else:
            x = self.conv_mix(x, edge_index)

        return x


class KipfNetd2(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh1=2, nh2=2, K=2, K_mix=1,
                 cached=True, inout_skipconn=True):
        super(KipfNetd2, self).__init__()
        self.inout_skipconn = inout_skipconn
        self.Kipfblock1 = Kipfblock(n_input=num_features, n_hidden=nh1, K=K)
        self.Kipfblock2 = Kipfblock(n_input=nh1, n_hidden=nh2, K=K)

        if inout_skipconn:
            self.conv_mix = ChebConv(nh2 + num_features, num_classes, K=K_mix)
        else:
            self.conv_mix = ChebConv(nh2, num_classes, K=K_mix)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.Kipfblock1(x, edge_index)
        x = self.Kipfblock2(x, edge_index)

        if self.inout_skipconn:
            x = torch.cat((x, data.x), 1)
            x = self.conv_mix(x, edge_index)
        else:
            x = self.conv_mix(x, edge_index)

        return x

class Graph_resnet(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh=38, K=6, K_mix=2,
                 inout_skipconn=True, depth=3, p=0.5, bn=False):
        super(Graph_resnet, self).__init__()
        self.inout_skipconn = inout_skipconn
        self.depth = depth

        self.Kipfblock_list = nn.ModuleList()
        self.skipproject_list = nn.ModuleList()

        if isinstance(nh, list):
            # if you give every layer a differnt number of channels
            # you need one number of channels for every layer!
            assert len(nh) == depth

        else:
            channels = nh
            nh = []
            for i in range(depth):
                nh.append(channels)

        for i in range(depth):
            if i == 0:
                self.Kipfblock_list.append(Kipfblock(n_input=num_features,
                                                     n_hidden=nh[0], K=K, p=p, bn=bn))
                self.skipproject_list.append(ChebConv(num_features, nh[0], K=1))
            else:
                self.Kipfblock_list.append(Kipfblock(n_input=nh[i - 1],
                                                     n_hidden=nh[i], K=K, p=p, bn=bn))
                self.skipproject_list.append(ChebConv(nh[i - 1], nh[i], K=1))

        if inout_skipconn:
            self.conv_mix = ChebConv(nh[-1] + num_features, num_classes, K=K_mix)
        else:
            self.conv_mix = ChebConv(nh[-1], num_classes, K=K_mix)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.depth):
            x = self.Kipfblock_list[i](x, edge_index) + \
                self.skipproject_list[i](x, edge_index)

        if self.inout_skipconn:
            x = torch.cat((x, data.x), 1)
            x = self.conv_mix(x, edge_index)
        else:
            x = self.conv_mix(x, edge_index)

        return x