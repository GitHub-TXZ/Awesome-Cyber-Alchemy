import matplotlib.pyplot as plt
import numpy as np
import torch
from opt1 import *
from EV_GCN_abs_L import EV_GCN
from utils.metrics import accuracy, auc, prf, save, auc_la,sa,kappa,to_icc,Over_all,compute_confidence_interval
from dataloader import dataloader
import warnings
warnings.filterwarnings("ignore")
import xlrd
import xlwt
import torch.nn.functional as F
from xlutils.copy import copy
import os
import sys
path_local = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path_local)
def write_excel_xls(path, sheet_name, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
    workbook.save(path)  # 保存工作簿
def write_excel_xls_append(path, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
    new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i + rows_old, j, value[i][j])  # 追加写入数据，注意是从i+rows_old行开始写入
    new_workbook.save(path)  # 保存工作簿
    print("xls格式表格【追加】写入数据成功！")
if __name__ == '__main__':
    a = 1
    proportion1 = 0.15
    named = "外部验证1"
    num_edges = np.zeros(5, dtype=np.float32)
    l = 0
    if a == 0:
        for asdw in [0.5]:
            Over_all(named, asdw)
    else:
        for proportion1 in [0.1,0.15,0.20,0.25,0.3,0.35]:#0.1,0.15,0.20,0.25,0.3,0.35,0.55,0.65,0.7,0.9    0.45,0.75,0.85,0.4,0.6
            for asdw in [0.55]:
                book_name_xls = f'外部验证 proportion={asdw}.xls'
                sheet_name_xls = 'xls格式表12'
                value_title = [["Model", "融合方法", "是否数据加强", "ACC", "ACC_std", "AUC", "AUC_std", "Sensitivity",
                                "Sensitivity_std", "Specificity", "Specificity_std", "precision", "precision_std",
                                "recall", "recall_std", "f1", "f1_std", "kappa", "kappa_std", "icc", "icc_std"],]
                write_excel_xls(book_name_xls, sheet_name_xls, value_title)
                lr = {"M1": 0.1, "M2": 0.1, "M3": 0.1, "M4": 0.1, "M5": 0.1, "M6": 0.1, "I": 0.1, "C": 0.1, "L": 0.1, "IC": 0.1, "ALL":0.1}
                edropout = {"M1": 0.4, "M2": 0.3, "M3": 0.4, "M4": 0.4, "M5": 0.4, "M6": 0.3, "I": 0.4, "C": 0.4, "L": 0.4,
                            "IC": 0.4, "ALL":0.4}
                dropout = {"M1": 0.1, "M2": 0.1, "M3": 0.01, "M4": 0.01, "M5": 0.1, "M6": 0.1, "I": 0.1, "C": 0.1, "L": 0.1,
                           "IC": 0.1,"ALL":0.1}
                num_iter = {"M1": 900, "M2": 900, "M3": 900, "M4": 900, "M5": 900, "M6": 900, "I": 900, "C": 900, "L": 900,
                            "IC": 900,"ALL":900}
                regions = ["M1"]#"M1","M2", "M3", "M4", "M5", "M6", "L", "I","C",
                #"M2", "M3", "M4", "M5", "M6", "L", "I","C"
                regions1 = ["ALL"]
                xy = [1]
                regions_edge = []
                for da in [0]:
                  for region in regions:
                      for k in xy:
                          if da == 1 or da == 0:
                            opt = OptInit().initialize()
                            print('  Loading dataset ...')
                            dl = dataloader()
                            disease, sub, y, L, R, ids = dl.load_data(region) # imaging features (raw), labels, non-image data+
                            x = y
                            n_folds = 10
                            edge_nums = np.zeros(n_folds, dtype=np.int32)
                            cv_splits = dl.data_split(n_folds)
                            corrects = np.zeros(n_folds, dtype=np.int32)
                            accs = np.zeros(n_folds, dtype=np.float32)
                            aucs = np.zeros(n_folds, dtype=np.float32)
                            prfs = np.zeros([n_folds, 3], dtype=np.float32)
                            sens = np.zeros(n_folds, dtype=np.float32)
                            spes = np.zeros(n_folds, dtype=np.float32)
                            loss_test = np.zeros(5, dtype=np.float32)
                            loss_train = np.zeros(5, dtype=np.float32)
                            acc_test_tu = np.zeros(5, dtype=np.float32)
                            kas = np.zeros(n_folds, dtype=np.float32)
                            iccs = np.zeros(n_folds, dtype=np.float32)
                            val_acc_best = 0
                            val_spe_best = 0
                            val_sen_best = 0
                            val_auc_best = 0
                            for fold in range(n_folds):
                                print("\r\n========================== Fold {},{} ==========================".format(fold,region))
                                train_ind = cv_splits[fold][0]  # cv_splits n * 2  返回索引
                                test_ind1 = cv_splits[fold][1]
                                test_ids = ids[test_ind1]
                                train_ids = ids[train_ind]
                                print('  Constructing graph data...')
                                node_ftr_dis, node_ftr_hea, node_ftr_all, y, train_ind, test_ind, val_ind = dl.get_node_features_load(k,region,da,fold)
                                post_ind, nega_ind = dl.post_nega(train_ind, y)
                                a = sum(y[train_ind])
                                b = len(y[train_ind])
                                c = (b - a) / a
                                data = 1
                                #a = sum(L[train_ind])
                                #b = len(L[train_ind])
                                #c1 = (b - a) / a
                                weights = torch.tensor([1, c], dtype=torch.float).to(opt.device)
                                #weights1 = torch.tensor([1, c1], dtype=torch.float).to(opt.device)
                                edge_index, edgenet_input, edge_labels, train_labels, edge_mask_numpy = dl.get_PAE_inputs(region,train_ind)
                                num_edge = edge_index.shape[1]
                                edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)
                                model = EV_GCN(node_ftr_dis.shape[1], opt.num_classes, dropout[region],
                                               edge_dropout=edropout[region], hgc=opt.hgc,
                                               lg=opt.lg, edgenet_input_dim=edgenet_input.shape[1] // 2, lg1=opt.lg1,
                                               gl=asdw)
                                model = model.to(opt.device)
                                if da == 0:
                                    loss_fn = torch.nn.CrossEntropyLoss()
                                    loss_fn1 = torch.nn.CrossEntropyLoss()
                                else:
                                    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
                                    loss_fn1 = torch.nn.CrossEntropyLoss()
                                    #loss_fn1 = torch.nn.CrossEntropyLoss(weight=weights1)
                                optimizer = torch.optim.Adam(model.parameters(), lr=lr[region], weight_decay=opt.wd)
                                features_cuda_dis = torch.tensor(node_ftr_dis, dtype=torch.float32).to(opt.device)
                                features_cuda_hea = torch.tensor(node_ftr_hea, dtype=torch.float32).to(opt.device)
                                edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
                                edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(opt.device)
                                labels = torch.tensor(y, dtype=torch.long).to(opt.device)
                                edge_mask = torch.tensor(edge_mask_numpy, dtype=torch.long).to(opt.device)
                                labels_L = torch.tensor(L, dtype=torch.long).to(opt.device)
                                edge_labels = torch.tensor(edge_labels, dtype=torch.long).to(opt.device)
                                one_shot = F.one_hot(edge_labels)
                                fold_model_path = opt.ckpt_path + "/discuss_fold{}_region_{}_shaibian_SAGcnwaibuceshi_{}.pth".format(fold,region,asdw)
                                n_samples = disease.shape[0]
                                def train():
                                    print("  Number of training samples %d" % len(train_ind))
                                    print("  Start training...\r\n")
                                    f1 = 0
                                    f1_val = 0
                                    loss_train_list = []
                                    loss_test_list = []
                                    epoch_list = []
                                    test_acc_list = []
                                    aver_list = []
                                    c = 0
                                    for epoch in range(num_iter[region]):
                                        model.train()
                                        optimizer.zero_grad()
                                        with torch.set_grad_enabled(True):
                                            node_logits, edge_weights, LR_logit, val,edge_in, edge_weight1 = model(features_cuda_dis, features_cuda_hea, edge_index, edgenet_input, edge_mask)
                                            num_size = int(len(train_ind) * proportion1)
                                            topk_index = dl.find_k_largest_indices(num_size, train_ind, test_ind,val_ind, val)
                                            #LR_loss = loss_fn1(LR_logit[train_ind], labels_L[train_ind])
                                            loss_cls = 0.9 * loss_fn(node_logits[train_ind], labels[train_ind]) + \
                                                       0.1 * loss_fn(node_logits[topk_index], labels[topk_index])

                                            loss = 0.9 * loss_cls + \
                                                   0.1 * dl.nn_loss(edge_weights, one_shot, edge_mask_numpy[topk_index], train_labels)
                                            loss.backward()
                                            optimizer.step()
                                        correct_train, acc_train,spe, sen,spe_num, sen_num = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])
                                        model.eval()
                                        with torch.set_grad_enabled(False):
                                            node_logits,edge_weights,LR_logit,_,edge_in, edge_weight1 = model(features_cuda_dis,features_cuda_hea, edge_index, edgenet_input,edge_mask)
                                            if epoch % 5 == 0 and epoch != 0:
                                                loss_test_list.append(np.mean(loss_test))
                                                loss_train_list.append(np.mean(loss_train))
                                                test_acc_list.append(np.mean(acc_test_tu))
                                                epoch_list.append(c)
                                                c += 1
                                            a = epoch % 5
                                            logits_test = node_logits[test_ind].detach().cpu().numpy()
                                            logits_val = node_logits[val_ind].detach().cpu().numpy()
                                            correct_test, acc_test, spe, sen, spe_num, sen_num = accuracy(logits_test, y[test_ind])
                                            correct_val, acc_val, spe_val, sen_val, spe_num_val, sen_num_val = accuracy(logits_val,
                                                                                                          y[val_ind])
                                            acc_test_tu[a] = acc_test
                                        auc_test = auc(logits_test, y[test_ind])
                                        auc_val = auc(logits_val, y[val_ind])
                                        prf_test = prf(logits_test, y[test_ind])
                                        prf_val = prf(logits_val, y[val_ind])
                                        print("Epoch: {},\tce loss: {:.5f},\ttrain acc: {:.5f},\ttest acc: {:.5f}, \tspe: {:.5f}, \tsen {:.5f}, \tval acc: {:.5f}, \tspe_val: {:.5f},\tsen_val: {:.5f}".
                                        format(epoch, loss.item(), acc_train.item(),acc_test.item(),spe, sen, acc_val.item(), spe_val, sen_val))
                                        if (prf_test[2] > f1 and epoch > 9) or (prf_test[2] == f1 and aucs[fold] < auc_test and epoch > 9):
                                            print("save!")
                                            f1 = prf_test[2]
                                            acc = acc_test
                                            aucs[fold] = auc_test
                                            corrects[fold] = correct_test
                                            accs[fold] = acc_test
                                            prfs[fold] = prf_test
                                            spes[fold] = spe_num
                                            sens[fold] = sen_num
                                            val_acc_best = acc_val
                                            val_sen_best = sen_val
                                            val_spe_best = spe_val
                                            f1_val = prf_val[2]
                                            val_auc_best = auc_val
                                            if opt.ckpt_path != '':
                                                if not os.path.exists(opt.ckpt_path):
                                                    os.makedirs(opt.ckpt_path)
                                                torch.save(model.state_dict(), fold_model_path)
                                    n = len(epoch_list)
                                    print("\r\n => Fold {} test accuacry {:.5f},auc {:.5f},f1 {:.5f} "
                                          "val accuacry {:.5f},auc_val {:.5f},f1_val {:.5f} ".format(
                                        fold, accs[fold],aucs[fold],f1, val_acc_best, val_auc_best, f1_val))
                                def evaluate():
                                    print("  Number of testing samples %d" % len(test_ind))
                                    print('  Start testing...')
                                    model.load_state_dict(torch.load(fold_model_path))
                                    model.eval()
                                    node_logits,_,_,_,edge_in, edge_weight1 = model(features_cuda_dis,features_cuda_hea, edge_index, edgenet_input,edge_mask)
                                    edge_nums[fold] = len(edge_weight1)
                                    logits_test = node_logits[test_ind].detach().cpu().numpy()
                                    save(logits_test, y[test_ind], local=ids[test_ind], named=named, region=region,
                                         asdw=asdw)
                                    print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))
                                #opt.train = 1
                                if opt.train == 1:
                                    train()
                                elif opt.train == 0:
                                    evaluate()
                            print("\r\n========================== Finish ==========================",region)
                            value = []#10
                            value.append(region)
                            value.append(str(k))#value_title = [["model", "融合方法", "是否数据加强", "ACC", "AUC","Sensitivity","Specificity","pr","rec","f1"],]
                            value.append(da)
                            n_samples = disease.shape[0]
                            print(np.sum(corrects))
                            acc_nfold = np.sum(corrects) / (n_samples)
                            value.append(str(acc_nfold))
                            value.append(str(np.std(accs)))
                            print("=> Average test accuracy in {}-fold CV: {:.5f}".format(n_folds, acc_nfold))
                            print("=> Average test AUC in {}-fold CV: {:.5f}".format(n_folds, np.mean(aucs)))
                            print("=> Average test edge in {}-fold CV: {:.5f}".format(n_folds, np.mean(edge_nums)/257))
                            regions_edge.append(np.mean(edge_nums)/257)
                            se, sp, f1 = np.mean(prfs, axis=0)
                            se_std, sp_std, f1_std = np.std(prfs, axis=0)
                            print("=> Average test sensitivity {:.4f}, specificity {:.4f}, F1-score {:.4f}".format(
                                np.mean(sens),
                                np.mean(spes), f1))

                            value.append(str(np.mean(aucs)))
                            value.append(str(np.std(aucs)))
                            value.append(str(np.mean(sens)))
                            value.append(str(np.std(sens)))
                            value.append(str(np.mean(spes)))
                            value.append(str(np.std(spes)))
                            value.append(str(se))
                            value.append(str(se_std))
                            value.append(str(sp))
                            value.append(str(sp_std))
                            value.append(str(f1))
                            value.append(str(f1_std))
                            value.append(str(np.mean(kas)))
                            value.append(str(np.std(kas)))
                            value.append(str(np.mean(iccs)))
                            value.append(str(np.std(iccs)))
                            value1 = []
                            value1.append(value)
                            write_excel_xls_append(book_name_xls, value1)
                print(asdw)
                print(sum(regions_edge)/len(regions_edge))
                print("置信区间",compute_confidence_interval(regions_edge))

