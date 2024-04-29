import torch
from torch.nn import Linear as Lin, Sequential as Seq
import torch_geometric as tg
import torch.nn.functional as F
from torch import nn
from PAE import PAE
from opt1 import *
import math
opt = OptInit().initialize()
class CKSA(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, edgenet_input_dim, edge_dropout, hgc, lg,lg1,gl=0.5):
        super(CKSA, self).__init__()
        K = 6
        hidden = [hgc for i in range(lg)]
        hidden1 = [hgc for i in range(lg1)]
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        bias = False
        self.gl = gl
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg1 = lg1
        self.gconv_LR = nn.ModuleList()
        for i in range(lg1):
            in_channels = input_dim if i == 0 else hidden1[i - 1]
            self.gconv_LR.append(tg.nn.ChebConv(int(in_channels), hidden[i], K, normalization='sym', bias=bias))
        cls_input_dim = sum(hidden)
        self.cls = nn.Sequential(
            torch.nn.Linear(cls_input_dim, 256),
            torch.nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            torch.nn.Linear(256, num_classes)
        )
        self.edge_net = PAE(input_dim=edgenet_input_dim, dropout=dropout)
        self.model_init()
        self.BD1 = BD()
    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
    def Select_anchor(self, edge_weight, mask, val, threshold):
        for i in range(val.shape[0]):
            cas = edge_weight[mask[i]] > threshold
            val[i] = torch.sum(edge_weight[mask[i]][cas])
        return val
    def step(self,index, weight):
        adj_matrix_sparse = torch.sparse_coo_tensor(index, weight, (self.n, self.n))
        A = adj_matrix_sparse.to_dense()
        A = torch.where(A >= self.gl, torch.tensor(1).to(opt.device), torch.tensor(0).to(opt.device))
        A = A.float()
        B = torch.matrix_power(A, self.K)
        C = 1 / (self.K * self.K) * B
        val_k = 2 / torch.sum(C, dim=1)
        val_1 = 2 / torch.sum(A, dim=1)
        return val_1 + val_k
    def forward(self, features_dis,features_hea, edge_index, edgenet_input, mask, enforce_edropout=False):
        threshold = self.gl
        edge_weight = torch.squeeze(self.edge_net(edgenet_input))
        tensor_zeros = torch.zeros(features_hea.shape[0])
        val = self.Select_anchor(edge_weight, mask, tensor_zeros, threshold)
        edge_weight_post = 1 - edge_weight
        merged_tensor = torch.stack((edge_weight_post, edge_weight), dim=1)
        edge_weight01 = (edge_weight > threshold).long()
        mask = torch.eq(edge_weight01, 1)
        edge_index, edge_weight = edge_index[:, mask], edge_weight[mask]
        if self.edge_dropout > 0:
            if enforce_edropout or self.training:
                one_mask_dis = torch.ones([edge_weight.shape[0], 1]).to(opt.device)
                self.drop_mask_dis = F.dropout(one_mask_dis, self.edge_dropout, True)
                self.bool_mask_dis = torch.squeeze(self.drop_mask_dis.type(torch.bool))
                edge_index = edge_index[:, self.bool_mask_dis]
                edge_weight = edge_weight[self.bool_mask_dis]

        #左侧特征
        features_dis = F.dropout(features_dis, self.dropout, self.training)
        h_dis = self.relu(self.gconv_LR[0](features_dis, edge_index, edge_weight))  #
        h0_dis = h_dis
        for i in range(1, self.lg1):
            h_dis = F.dropout(h_dis, self.dropout, self.training)
            h_dis = self.relu(self.gconv_LR[i](h_dis, edge_index, edge_weight))
            jk = torch.cat((h0_dis, h_dis), dim=1)
            h0_dis = jk

        #右侧特征
        features_hea = F.dropout(features_hea, self.dropout, self.training)
        h_hea = self.relu(self.gconv_LR[0](features_hea,edge_index, edge_weight))
        h0_hea = h_hea
        for i in range(1, self.lg1):
            h_hea = F.dropout(h_hea, self.dropout, self.training)
            h_hea = self.relu(self.gconv_LR[i](h_hea, edge_index, edge_weight))
            jk = torch.cat((h0_hea,h_hea), dim=1)
            h0_hea = jk
        chayi = torch.abs(h0_dis - h0_hea)
        #stroke = self.BD1(chayi, h0_dis, h0_hea)
        #logit = self.cls(torch.cat((h0_hea), dim=1))
        logit = self.cls(h0_hea)
        return logit,merged_tensor, 1, val

class BD(torch.nn.Module):
    def __init__(self,):
        super(BD, self).__init__() #X_stroke = a * X_left + b * X_right  求a, b
        self.left = nn.Sequential(
            nn.Linear(80, 160),
        nn.BatchNorm1d(160)
        )
        self.right = nn.Sequential(
            nn.Linear(80, 160),
            nn.BatchNorm1d(160)
        )
        self.sub = nn.Sequential(
            nn.Linear(80, 160),
            nn.BatchNorm1d(160)
        )
    def forward(self, sub, left, right):
        R_sub = self.sub(sub)
        R_left = self.left(left)
        R_right = self.right(right)
        t = R_right.t()
        sub_left = torch.diag(torch.matmul(R_sub, R_left.t()))
        sub_right = torch.diag(torch.matmul(R_sub, t))
        left_right = torch.diag(torch.matmul(R_left, t))
        all_sim = torch.cat((sub_left.unsqueeze(0), sub_right.unsqueeze(0), left_right.unsqueeze(0)), dim=0)
        all_pos = F.softmax(all_sim, dim=0).t()
        x_storke = all_pos[:, 0].unsqueeze(1) * left + all_pos[:, 1].unsqueeze(1) \
                   * right + all_pos[:, 2].unsqueeze(1) * sub
        return x_storke


