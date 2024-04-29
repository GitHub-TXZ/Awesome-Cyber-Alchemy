import os.path

import pandas as pd

import data.ABIDEParser as Reader
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
import random
from data import batchprocessing as ba
from imblearn.over_sampling import  ADASYN
import copy
import scipy.io as sio
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
index = {"M1": 0, "M2": 1, "M3": 2, "M4": 3, "M5": 4, "M6": 5, "I": 6, "C": 7, "L": 8, "IC": 9, "ALL":10}
fnum = {"M1": 200, "M2": 200, "M3": 250, "M4": 250, "M5": 300, "M6": 250, "I": 250, "C": 300, "L": 250, "IC": 300}
score = {"M1": 0.70, "M2": 0.65, "M3": 0.75, "M4": 0.65, "M5": 0.70, "M6": 0.70, "I": 0.65, "C": 0.70, "L": 0.70, "IC": 0.65}
class dataloader():
    def __init__(self): 
        self.pd_dict = {}
        self.node_ftr_dim = 2000
        self.num_classes = 2
        self.dis_lis = []
        self.sub_lis = []
        self.hea_lis = []
        self.y_lis = []

    def load_data(self,region):
        ''' load multimodal data from ABIDE
        return: imaging features (raw), labels, non-image data
        "/homeb/lining/Data/ProveIT_ASPECTS_Data/train_test_select.txt"

        '''

        subject_IDs = Reader.get_ids(path="/homeb/lining/Data/KSR_bag/subjects.txt")
        self.subject_IDs = subject_IDs
        self.y_M1 = Reader.get_subject_lable(subject_IDs, "M1")
        self.y_lis.append(self.y_M1)
        self.y_M2 = Reader.get_subject_lable(subject_IDs, "M2")
        self.y_lis.append(self.y_M2)
        self.y_M3 = Reader.get_subject_lable(subject_IDs, "M3")
        self.y_lis.append(self.y_M3)
        self.y_M4 = Reader.get_subject_lable(subject_IDs, "M4")
        self.y_lis.append(self.y_M4)
        self.y_M5 = Reader.get_subject_lable(subject_IDs, "M5")
        self.y_lis.append(self.y_M5)
        self.y_M6 = Reader.get_subject_lable(subject_IDs, "M6")
        self.y_lis.append(self.y_M6)
        self.y_I = Reader.get_subject_lable(subject_IDs, "I")
        self.y_lis.append(self.y_I)
        self.y_C = Reader.get_subject_lable(subject_IDs, "C")
        self.y_lis.append(self.y_C)
        self.y_L = Reader.get_subject_lable(subject_IDs, "L")
        self.y_lis.append(self.y_L)
        self.y_IC = Reader.get_subject_lable(subject_IDs, "IC")
        self.y_lis.append(self.y_IC)
        self.y_all = Reader.get_subject_lable(subject_IDs, f"Bleed6")
        self.y_lis.append(self.y_all)
        self.M1_dis, self.M1_hea = Reader.get_networks(subject_IDs, "M1")
        self.dis_lis.append(self.M1_dis)
        self.hea_lis.append(self.M1_hea)
        self.M2_dis, self.M2_hea = Reader.get_networks(subject_IDs, "M2")
        self.dis_lis.append(self.M2_dis)
        self.hea_lis.append(self.M2_hea)
        self.M3_dis, self.M3_hea = Reader.get_networks(subject_IDs, "M3")
        self.dis_lis.append(self.M3_dis)
        self.hea_lis.append(self.M3_hea)
        self.M4_dis, self.M4_hea = Reader.get_networks(subject_IDs, "M4")
        self.dis_lis.append(self.M4_dis)
        self.hea_lis.append(self.M4_hea)
        self.M5_dis, self.M5_hea = Reader.get_networks(subject_IDs, "M5")
        self.dis_lis.append(self.M5_dis)
        self.hea_lis.append(self.M5_hea)
        self.M6_dis, self.M6_hea = Reader.get_networks(subject_IDs, "M6")
        self.dis_lis.append(self.M6_dis)
        self.hea_lis.append(self.M6_hea)
        self.I_dis, self.I_hea = Reader.get_networks(subject_IDs, "I")
        self.dis_lis.append(self.I_dis)
        self.hea_lis.append(self.I_hea)
        self.C_dis, self.C_hea= Reader.get_networks(subject_IDs, "C")
        self.dis_lis.append(self.C_dis)
        self.hea_lis.append(self.C_hea)
        self.L_dis, self.L_hea = Reader.get_networks(subject_IDs, "L")
        self.dis_lis.append(self.L_dis)
        self.hea_lis.append(self.L_hea)
        self.IC_dis, self.IC_hea= Reader.get_networks(subject_IDs, "IC")
        self.dis_lis.append(self.IC_dis)
        self.hea_lis.append(self.IC_hea)
        self.IC_dis, self.IC_hea = Reader.get_networks(subject_IDs, "IC")
        self.dis_lis.append(self.IC_dis)
        self.hea_lis.append(self.IC_hea)
        self.Right = Reader.get_subject_lable(subject_IDs, "Right")
        self.Left = Reader.get_subject_lable(subject_IDs, "Left")
        self.disease, self.sub, self.y = copy.copy(self.dis_lis[index[region]]),copy.copy(self.hea_lis[index[region]]),copy.copy(self.y_lis[index[region]])
        return self.disease, self.sub, self.y, self.Left, self.Right, subject_IDs

    def abb(self, A, B):
        x_data = list()
        for i in range(A.shape[0]):
            arr = np.concatenate((A[i], B[i]))
            x_data.append(arr)
        x_data = np.array(x_data)
        return x_data
    def data_split(self, n_folds):
        # split data by k-fold CV
        skf = StratifiedKFold(n_splits=n_folds)
        cv_splits = list(skf.split(self.disease, self.y))
        return cv_splits
    def post_nega(self,train_ind,y):
        post_ind = []
        nega_ind = []
        for i in train_ind:
            if y[i] == 0.0:
                nega_ind.append(i)
            if y[i] == 1.0:
                post_ind.append(i)
        post_ind = np.asarray(post_ind)
        nega_ind = np.asarray(nega_ind)
        return post_ind, nega_ind
    def forest(self, feature1, labels, train_ind, fnum):
        feature1 = ba.normalization1(feature1)
        train_dis, y_dis = feature1[train_ind],labels[train_ind]
        rfc = RandomForestClassifier()
        num_estimator = {'n_estimators': range(50, 400, 50)}  # 随机森林中树的棵数，以50为起点，50为步长，最多为300棵树
        gs1 = GridSearchCV(estimator=rfc, param_grid=num_estimator, scoring='roc_auc', cv=3)
        gs1.fit(train_dis, y_dis)
        maxdepth = {'max_depth': range(3, 10, 1)}
        gs2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=gs1.best_estimator_.n_estimators),
                           param_grid=maxdepth, scoring='roc_auc', cv=3)
        gs2.fit(train_dis, y_dis)
        minsamples = {'min_samples_split': range(2, 50, 2)}
        gs3 = GridSearchCV(estimator=RandomForestClassifier(max_depth=gs2.best_estimator_.max_depth,
                                                            n_estimators=gs1.best_estimator_.n_estimators),
                           param_grid=minsamples, scoring='roc_auc', cv=3)
        gs3.fit(train_dis, y_dis)
        best_rfc = RandomForestClassifier(max_depth=gs2.best_estimator_.max_depth,
                                          min_samples_split=gs3.best_estimator_.min_samples_split,
                                          n_estimators=gs1.best_estimator_.n_estimators)  # 使用最优参数对随机森林进行类的实例化
        best_rfc.fit(train_dis, y_dis)  # 模型拟合
        importance1 = best_rfc.feature_importances_
        sorted_id = sorted(range(len(importance1)), key=lambda k: importance1[k], reverse=True)
        feature2 = feature1[:,sorted_id]
        feature2 = feature2[:,:fnum]
        return feature2

    def select(self,train_ind,k,region):
        fnum1 = {"M1": 15, "M2": 20, "M3": 15, "M4": 20, "M5": 25, "M6": 20, "I": 20, "C": 25, "L": 20, "IC": 25}
        atten_dis = self.forest(self.dis_lis[index[region]], self.y_lis[index[region]], train_ind, fnum[region])
        atten_hea = self.forest(self.hea_lis[index[region]], self.y_lis[index[region]], train_ind, fnum[region])
        if k == 0:
            return atten_dis,atten_hea
        if k == 1:
            els = {"M1": [1,3,6], "M2": [0,2,4,6], "M3": [1,5], "M4": [0,6,5], "M5": [1,3,5,6],
                   "M6": [2,4], "I": [0,1,3,4,8], "C": [8,9], "L": [6,7],
                    "IC": [7,8]}
            for i in els[region]:
                atten_dis = self.abb(self.forest(self.dis_lis[i],self.y_lis[i],train_ind,fnum1[region]),atten_dis)
                atten_hea = self.abb(self.forest(self.hea_lis[i], self.y_lis[i], train_ind, fnum1[region]), atten_hea)
            return atten_dis, atten_hea
        if k == 2:
            els = {"M1": [1,4,3,6,8], "M2": [4,6], "M3": [1,4,5,6], "M4": [1,4,0,5,6,8], "M5": [1,6],
                   "M6": [1,2,4,6], "I": [1,4,8], "C": [1,4,6,8], "L": [6,7],
                    "IC": [4,6,8,7]}
            for i in els[region]:
                atten_dis = self.abb(self.forest(self.dis_lis[i], self.y_lis[i], train_ind, fnum1[region]), atten_dis)
                atten_hea = self.abb(self.forest(self.hea_lis[i], self.y_lis[i], train_ind, fnum1[region]), atten_hea)
            return atten_dis, atten_hea
    def get_node_features(self, train_ind, test_ind, k, region,da):
        '''
            preprocess node features for ev-gcn
        '''
        sam = 0.5
        if region == "M2" or region == "M5" or region == "I" or region == "L":
            sam = 1
        self.node_ftr_dis, self.node_ftr_hea = self.select(train_ind, k, region)
        if da == 0 or da == 1:
            return self.node_ftr_dis, self.node_ftr_hea, self.y, train_ind, test_ind
        else:
            return self.node_ftr_dis, self.node_ftr_hea, self.y, train_ind, test_ind
    def get_node_features_load(self,k, region, da, fold):
        base_path = f"/homeb/lining/Data/KSR_bag/SVM_LR/"
        base_path = os.path.join(base_path, region, str(fold))
        base_path_dis = os.path.join(base_path,"L.mat")
        self.node_ftr_dis = sio.loadmat(base_path_dis)["feature"]

        base_path_hea = os.path.join(base_path, "R.mat")
        self.node_ftr_hea = sio.loadmat(base_path_hea)["feature"]

        base_path_sub = os.path.join(base_path, "sub.mat")
        self.node_ftr_sub = sio.loadmat(base_path_sub)["feature"]

        tran_ind = pd.read_csv(os.path.join(base_path, "train.csv"),header=None).values.flatten()
        test_ind = pd.read_csv(os.path.join(base_path, "test.csv"),header=None).values.flatten()
        self.node_ftr_all = self.node_ftr_sub
        return self.node_ftr_dis,self.node_ftr_hea,self.node_ftr_all, self.y, tran_ind, test_ind

    def get_node_features_load_pro(self, k, region, da, fold):
        base_path = f"/homeb/lining/lining_4/Data/ProveIT_ASPECTS_Data/SVM_remove"
        base_path = os.path.join(base_path, region, str(fold))
        train_ind = pd.read_csv(os.path.join(base_path, "train.csv"), header=None).values.flatten()

        base_path_dis = os.path.join(base_path, "hea.mat")
        node_ftr_dis = sio.loadmat(base_path_dis)["feature"][train_ind]

        base_path_hea = os.path.join(base_path, "dis.mat")
        node_ftr_hea = sio.loadmat(base_path_hea)["feature"][train_ind]

        base_path_sub = os.path.join(base_path, "sub.mat")
        node_ftr_sub = sio.loadmat(base_path_sub)["feature"][train_ind]
        labels = self.y_pro[train_ind]
        node_ftr_all = node_ftr_sub
        return node_ftr_dis, node_ftr_hea, node_ftr_all, labels

    def get_node_features_load_yanzheng(self, k, region, da, fold):
        base_path = f"/homeb/lining/lining_4/Data/ProveIT_ASPECTS_Data/SVM_remove"
        base_path = os.path.join(base_path, region, "0")
        test_ind = pd.read_csv(os.path.join(base_path, "test.csv"), header=None).values.flatten()

        base_path_dis = os.path.join(base_path,"hea.mat")
        node_ftr_dis = sio.loadmat(base_path_dis)["feature"][test_ind]

        base_path_hea = os.path.join(base_path, "dis.mat")
        node_ftr_hea = sio.loadmat(base_path_hea)["feature"][test_ind]

        base_path_sub = os.path.join(base_path, "sub.mat")
        node_ftr_sub = sio.loadmat(base_path_sub)["feature"][test_ind]
        labels = self.y_pro[test_ind]
        node_ftr_all = node_ftr_sub
        return node_ftr_dis,node_ftr_hea,node_ftr_all, labels
    def Data_Aug(self, train_ind, test_ind,sam):
        train = self.node_ftr[train_ind]
        test = self.node_ftr[test_ind]
        train_label = self.y[train_ind]
        test_label = self.y[test_ind]
        A = ADASYN(sampling_strategy=sam)
        train, train_label = A.fit_resample(train, train_label)
        t = len(train_label)
        index = [i for i in range(len(train))]
        random.shuffle(index)
        train = train[index]
        train_label = train_label[index]  # 打乱数据集
        train = np.vstack((train, test))
        train_label = np.hstack((train_label, test_label))
        te = len(train_label)
        train_ind = np.array([i for i in range(t)])
        test_ind = np.array([i for i in range(t, te)])
        self.node_ftr = train
        return train_ind, test_ind, train_label

    def get_PAE_inputs(self, regon, train_inds):
        '''
            get PAE inputs for ev-gcn
        '''
        # construct edge network inputs
        n = self.node_ftr_all.shape[0]  # 有多少个节点
        num_edge = n * (1 + n) // 2 - n  # 多少条边
        # pd_ftr_dim = nonimg.shape[1] # 3
        edge_mask = np.zeros([n, n], dtype=np.int64)
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        edge_labels = np.ones([num_edge], dtype=np.int64)
        edgenet_input = np.zeros([num_edge, self.node_ftr_all.shape[1] * 2], dtype=np.float32)
        aff_score = np.zeros(num_edge, dtype=np.float32)  # 每条边的得分
        # static affinity score used to pre-prune edges
        aff_adj = Reader.get_static_affinity_adj(self.node_ftr_all)  # 126 * 126 对称矩阵 代表i，j的距离
        nodes = self.node_ftr_all
        flatten_ind = 0
        labels_train = []
        for i in range(n):
            for j in range(i + 1, n):
                k = ((2 * n - i - 1) * i) / 2 + j - i - 1
                edge_mask[i][j] = k
                edge_mask[j][i] = k
                edge_index[:, flatten_ind] = [i, j]
                edgenet_input[flatten_ind] = np.concatenate((nodes[i], nodes[j]))
                aff_score[flatten_ind] = aff_adj[i][j]
                if self.y[i] != self.y[j]:
                    edge_labels[flatten_ind] = 0
                if i not in train_inds or j not in train_inds:
                    edge_labels[flatten_ind] = 0
                if i in train_inds and j in train_inds:
                    labels_train.append(flatten_ind)
                flatten_ind += 1
        assert flatten_ind == num_edge, "Error in computing edge input"
        labels_train = np.array(labels_train)
        keep_ind = np.where(aff_score > 0)[0]
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]
        return edge_index, edgenet_input, edge_labels, labels_train, edge_mask

    def nn_loss(self, pred_all, true_all,topk_index, train):
        pred_log_all = torch.log(pred_all)
        true = true_all[train]
        pred = pred_all[train]
        true1 = true.cpu().detach().numpy()
        pred1 = pred.cpu().detach().numpy()
        topk_index = topk_index.reshape(-1)
        indexes_ture = np.where(true1[:, 0] == 1)[0]  # 真实标签为0
        indexes_pred = np.where(pred1[:, 0] < 0.50)[0]  # 预测标签为1
        intersection = np.intersect1d(indexes_ture, indexes_pred)  # 真实标签为0预测为1，即应该没有边但模型给出了边
        pred_log = torch.log(pred)

        nllloss = (- torch.sum(true * pred_log)-
                   torch.sum(true_all[topk_index] * pred_log_all[topk_index])) / \
                  (pred.shape[0] + topk_index.shape[0])
        return nllloss
    def nn_loss_k(self, pred_all, true_all,topk_index, train):
        pred_log_all = torch.log(pred_all)
        true = true_all[train]
        pred = pred_all[train]
        true1 = true.cpu().detach().numpy()
        pred1 = pred.cpu().detach().numpy()
        topk_index = topk_index.reshape(-1)
        indexes_ture = np.where(true1[:, 0] == 1)[0]  # 真实标签为0
        indexes_pred = np.where(pred1[:, 0] < 0.50)[0]  # 预测标签为1
        intersection = np.intersect1d(indexes_ture, indexes_pred)  # 真实标签为0预测为1，即应该没有边但模型给出了边
        pred_log = torch.log(pred)

        nllloss = (- torch.sum(true * pred_log)-
                   torch.sum(true[intersection] * pred_log[intersection])) / \
                  (pred.shape[0] + intersection.shape[0])
        return nllloss
    def loss_anchor(self,true, pred):
        pred_log = torch.log(pred)
        nllloss = (- torch.sum(true * pred_log)) / \
                  (pred.shape[0])
        return nllloss

    def find_k_largest_indices(self, k, train, test, val_ind, val):
        if k <= 0:
            return []
        sorted_indices = torch.argsort(val).tolist()
        result = [x for x in sorted_indices if (x not in test) and (x not in val_ind)]
        result = result[:k]
        for i in test:
            if i in result:
                return [1, 2, 3]
        return result
    def subgraph(self,topk_index,index,weight,test_ind):
        a = 0.65 #控制子图节点数量
        index = index.detach().cpu().numpy()
        weight = weight.detach().cpu().numpy()
        indices = np.where(weight > a)
        if len(indices) == 0:
            return []
        index = index[indices]
        result = []
        for idx in topk_index:
            column_idx = -1
            for j in range(len(index[0])):
                if index[0][j] == idx:
                    column_idx = index[1][j]
                    result.append(column_idx)
                    #if weight[j] > a:
                    #    result.append(column_idx)
                elif index[1][j] == idx:
                    column_idx = index[0][j]
                    result.append(column_idx)
                    #if weight[j] > a:
                    #    result.append(column_idx)
                else:
                    pass
        result_list = list(set(result))
        new_listB = [elem for elem in result_list if elem not in test_ind]
        new_listB = [elem for elem in new_listB if elem not in topk_index]
        for i in test_ind:
            if i in new_listB:
                return [1, 2, 3]
        for i in topk_index:
            if i in new_listB:
                return [1, 2, 3]
        return new_listB







