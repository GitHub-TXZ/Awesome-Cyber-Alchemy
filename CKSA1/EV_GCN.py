import torch
from torch.nn import Linear as Lin, Sequential as Seq
import torch_geometric as tg
import torch.nn.functional as F
from torch import nn
from PAE import PAE
from opt import *
import math
opt = OptInit().initialize()

class EV_GCN(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, edgenet_input_dim, edge_dropout, hgc, lg,lg1,num,num_edge,gl=0.25,tq=0.5):
        super(EV_GCN, self).__init__()
        K = 6
        hidden = [hgc for i in range(lg)]
        hidden1 = [hgc for i in range(lg1)]
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        bias = False
        self.gl = gl
        self.tq = tq
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg1 = lg1
        self.gconv_LR = nn.ModuleList()
        for i in range(lg1):
            in_channels = input_dim if i == 0 else hidden1[i - 1]
            self.gconv_LR.append(tg.nn.ChebConv(int(in_channels), hidden[i], K, normalization='sym', bias=bias))
        cls_input_dim = sum(hidden) + sum(hidden1)
        out_dim = 64
        self.cls = nn.Sequential(
            torch.nn.Linear(out_dim*2, 256),
            torch.nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            torch.nn.Linear(256, num_classes)
        )
        self.endcoder_dis = nn.Sequential(
            torch.nn.Linear(cls_input_dim // 2, 512),
            torch.nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            torch.nn.Linear(512, out_dim)
        )
        self.endcoder_hea = nn.Sequential(
            torch.nn.Linear(cls_input_dim // 2, 512),
            torch.nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            torch.nn.Linear(512, out_dim)
        )
        self.edge_net = PAE(input_dim=edgenet_input_dim, dropout=dropout)
        self.model_init()
        self.matrix_dis = nn.Parameter(torch.randn(num_edge, 1), requires_grad=True)
    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def make_adj(self, edgenet_index, edge_weight, fea):
        n = fea.shape[0]
        edgenet_index = edgenet_index.detach().cpu().numpy()
        edge_weight = edge_weight.detach().cpu().numpy()
        adj = np.ones((n, n))
        i = 0
        for k, l in edgenet_index.T:
            adj[int(k)][int(l)] = edge_weight[i]
            adj[int(l)][int(k)] = edge_weight[i]
            i += 1
        return adj
    def make_edg(self,x,edge):
        x = x.detach().cpu().numpy()
        n = x.shape[0]  # 有多少个节点
        num_edge = n * (1 + n) // 2 - n  # 多少条边
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        edgenet_input = np.zeros([num_edge, 1], dtype=np.float32)
        aff_score = np.zeros(num_edge, dtype=np.float32)  # 每条边的得分
        flatten_ind = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[:, flatten_ind] = [i, j]  # 第flatten_ind列
                edgenet_input[flatten_ind] = (x[i][j] + x[j][i])/2   # con
                aff_score[flatten_ind] =(x[i][j] + x[j][i])/2
                flatten_ind += 1
        keep_ind = np.where(aff_score > edge)[0]
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]
        return edge_index, edgenet_input
    def cal_cosloss(self, a, b):
        c = torch.cosine_similarity(a, b, dim=1)
        d = torch.mean(c)
        e = torch.tensor(math.e, dtype=torch.float32).to(opt.device)
        loss = torch.pow(e, d)
        return loss
    def cal_loss(self, a, b, post_ind, nega_ind):
        tq = self.tq
        c = torch.cosine_similarity(a, b, dim=1)
        c = torch.exp(c / tq)
        post = torch.sum(c[post_ind])
        nega = torch.sum(c[nega_ind])
        loss = - torch.log(post / nega)
        return loss
    def graph_learn(self,index, weight):
        edge_weight = weight - weight * torch.squeeze(self.matrix_dis)
        norm = torch.linalg.norm(edge_weight, ord=2, ) + torch.linalg.norm(self.matrix_dis, ord=2, )
        idx = torch.nonzero(edge_weight > self.gl)
        edge_weight = torch.index_select(edge_weight, 0, idx.squeeze())
        index = torch.index_select(index, 1, idx.squeeze())
        return index, edge_weight, norm
    def kk(self):
        reg_loss = 0.0
        state_dict = self.endcoder_dis.state_dict()  # 获取模型参数字典
        for name, param in state_dict.items():
            if name == "3.weight":  # 判断是否是第二个全连接层的参数
                reg_loss += torch.sqrt(torch.sum(param ** 2))
        state_dict = self.endcoder_hea.state_dict()  # 获取模型参数字典
        for name, param in state_dict.items():
            if name == "3.weight":  # 判断是否是第二个全连接层的参数
                reg_loss += torch.sqrt(torch.sum(param ** 2))
        return reg_loss
    def forward(self, features_dis, edge_index_dis, edgenet_input_dis,features_hea, edge_index_hea, edgenet_input_hea, post_ind, nega_ind, enforce_edropout=False):
        edge_weight_dis = torch.squeeze(self.edge_net(edgenet_input_dis))
        edge_index_dis, edge_weight_dis, norm_dis = self.graph_learn(edge_index_dis, edge_weight_dis)
        if self.edge_dropout > 0:
            if enforce_edropout or self.training:
                one_mask_dis = torch.ones([edge_weight_dis.shape[0], 1]).to(opt.device)
                self.drop_mask_dis = F.dropout(one_mask_dis, self.edge_dropout, True)
                self.bool_mask_dis = torch.squeeze(self.drop_mask_dis.type(torch.bool))
                edge_index_dis = edge_index_dis[:, self.bool_mask_dis]
                edge_weight_dis = edge_weight_dis[self.bool_mask_dis]
        features_dis = F.dropout(features_dis, self.dropout, self.training)
        h_dis = self.relu(self.gconv_LR[0](features_dis, edge_index_dis, edge_weight_dis))  #
        h0_dis = h_dis
        for i in range(1, self.lg1):
            h_dis = F.dropout(h_dis, self.dropout, self.training)
            h_dis = self.relu(self.gconv_LR[i](h_dis, edge_index_dis, edge_weight_dis))
            jk = torch.cat((h0_dis, h_dis), dim=1)
            h0_dis = jk
        features_hea = F.dropout(features_hea, self.dropout, self.training)
        h_hea = self.relu(self.gconv_LR[0](features_hea,edge_index_dis, edge_weight_dis))
        h0_hea = h_hea
        for i in range(1, self.lg1):
            h_hea = F.dropout(h_hea, self.dropout, self.training)
            h_hea = self.relu(self.gconv_LR[i](h_hea, edge_index_dis, edge_weight_dis))
            jk = torch.cat((h0_hea,h_hea), dim=1)
            h0_hea = jk
        norm = norm_dis
        #loss1 = torch.dist(x_dis,x_hea, p=2)
        #loss2 = self.cal_cosloss(h0_dis,h0_hea)
        h0_dis1 = self.endcoder_dis(h0_dis)
        h0_hea1 = self.endcoder_hea(h0_hea)
        loss2 = self.cal_loss(h0_dis1, h0_hea1, post_ind, nega_ind)
        loss1 = self.kk()
        chayi = h0_dis1 - h0_hea1
        go = torch.cat((h0_dis1, chayi), dim=1)
        logit = self.cls(go)
        return logit,norm,loss2,loss1



