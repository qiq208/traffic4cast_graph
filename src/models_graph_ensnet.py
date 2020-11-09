import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import ChebConv, LayerNorm, SAGEConv, SGConv  # noqa

class Kipfblock(torch.nn.Module):
    def __init__(self, n_input, n_hidden=64, K=8, p=0.5, bn=False):
        super(Kipfblock, self).__init__()
        self.conv1 = ChebConv(n_input, n_hidden, K=K)
        self.p = p
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.do_bn = bn
        if bn:
            self.bn = LayerNorm(n_hidden)

    def forward(self, x, edge_index):
        if self.do_bn:
            x = F.elu(self.bn(self.conv1(x, edge_index)))
        else:
            x = F.elu(self.conv1(x, edge_index))

        #x = F.dropout(x, training=self.training, p=self.p)

        return x
    
class Sageblock(torch.nn.Module):
    def __init__(self, n_input, n_hidden, bn=False):
        super(Sageblock, self).__init__()
        self.conv1 = SAGEConv(n_input, n_hidden)
        #self.p = p
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.do_bn = bn
        if bn:
            self.bn = LayerNorm(n_hidden)

    def forward(self, x, edge_index):
        if self.do_bn:
            x = F.elu(self.bn(self.conv1(x, edge_index)))
        else:
            x = F.elu(self.conv1(x, edge_index))

        #x = F.dropout(x, training=self.training, p=self.p)
        return x

class Sgblock(torch.nn.Module):
    def __init__(self, n_input, n_hidden, K=3, bn=False):
        super(Sgblock, self).__init__()
        self.conv1 = SGConv(n_input, n_hidden, K, cached=False)
        #self.p = p
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.do_bn = bn
        if bn:
            self.bn = LayerNorm(n_hidden)

    def forward(self, x, edge_index):
        if self.do_bn:
            x = F.elu(self.bn(self.conv1(x, edge_index)))
        else:
            x = F.elu(self.conv1(x, edge_index))

        #x = F.dropout(x, training=self.training, p=self.p)
        return x
    
    
class Graph_ensnet(torch.nn.Module):
    def __init__(self, num_features, num_classes, nh=38, K=6, K_mix=2,
                 inout_skipconn=True, depth=3, p=0.5, bn=False):
        super(Graph_ensnet, self).__init__()
        self.inout_skipconn = inout_skipconn
        self.depth = depth

        self.Kipfblock_list = nn.ModuleList()
        self.Sage_list = nn.ModuleList()
        self.Sg_list = nn.ModuleList()
        #self.skipproject_list = nn.ModuleList()
        #self.norms_list = nn.ModuleList()

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
                self.Sage_list.append(Sageblock(num_features, nh[0]))
                self.Sg_list.append(Sgblock(num_features, nh[0], K=5))
                #self.skipproject_list.append(ChebConv(num_features, nh[0], K=1))
            else:
                self.Kipfblock_list.append(Kipfblock(n_input=nh[i - 1],
                                                     n_hidden=nh[i], K=K, p=p, bn=bn))
                self.Sage_list.append(Sageblock(nh[i - 1], nh[0]))
                self.Sg_list.append(Sgblock(nh[i - 1], nh[0], K=5))
                #self.skipproject_list.append(ChebConv(nh[i - 1], nh[i], K=1))

        if inout_skipconn:
            self.conv_mix = ChebConv(nh[-1] + num_features, num_classes, K=K_mix)
            self.sage_mix = Sageblock(nh[-1] + num_features, num_classes)
            self.sg_mix=Sgblock(nh[-1] + num_features, num_classes, K=5)
        else:
            self.conv_mix = ChebConv(nh[-1], num_classes, K=K_mix)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.depth):
            x = (self.Kipfblock_list[i](x, edge_index) + \
                self.Sage_list[i](x, edge_index) + \
                self.Sg_list[i](x, edge_index))/3
                #self.Sg_list[i](x, edge_index) + \
                #self.skipproject_list[i](x, edge_index))/4
                #F.elu(self.norms_list[i](self.skipproject_list[i](x, edge_index)))

        if self.inout_skipconn:
            x = torch.cat((x, data.x), 1)
            x = (self.conv_mix(x, edge_index) + self.sage_mix(x, edge_index) + self.sg_mix(x, edge_index))/3
        else:
            x = self.conv_mix(x, edge_index)

        return x