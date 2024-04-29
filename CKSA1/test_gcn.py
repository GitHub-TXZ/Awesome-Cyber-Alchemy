import os
import sys
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
import random
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from opt import *
from CKSA import EV_GCN
from utils.metrics import accuracy, auc, prf, save, auc_la
from dataloader import dataloader

if __name__ == '__main__':
    opt = OptInit().initialize()

    print('  Loading dataset ...')
    dl = dataloader()
    disease, sub, y = dl.load_data(r"/home/lining/GCN/CKSA/data/test.txt",r"/home/lining/GCN/CKSA/data/label_test.csv")  # imaging features (raw), labels, non-image data
    x = y
    n_folds = 10

    #cv_splits = dl.data_split(n_folds)
    # print(cv_splits)

    corrects = np.zeros(n_folds, dtype=np.int32)
    accs = np.zeros(n_folds, dtype=np.float32)
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds, 3], dtype=np.float32)
    spe = np.zeros(n_folds, dtype=np.float32)
    sen = np.zeros(n_folds, dtype=np.float)

    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold))
        test_ind1 = [i for i in range(0, 15)]
        train_ind = [i for i in range(15, 178)]

        print('  Constructing graph data...')
        # extract node features

        # node_ftr = dl.get_node_features(train_ind)   #降维2000
        # print(node_ftr.shape[0])
        node_ftr, y, train_ind, test_ind = dl.get_node_features_test(train_ind, test_ind1)

        # get PAE inputs
        edge_index, edgenet_input1 = dl.get_PAE_inputs()
        # normalization for PAE
        edgenet_input = (edgenet_input1 - edgenet_input1.mean(axis=0)) / edgenet_input1.std(axis=0)

        # build network architecture
        model = EV_GCN(node_ftr.shape[1], opt.num_classes, opt.dropout, edge_dropout=opt.edropout, hgc=opt.hgc,
                       lg=opt.lg, edgenet_input_dim=edgenet_input.shape[1] // 2).to(opt.device)
        model = model.to(opt.device)

        # build loss, optimizer, metric
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)

        features_cuda = torch.tensor(node_ftr, dtype=torch.float32).to(opt.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(opt.device)
        labels = torch.tensor(y, dtype=torch.long).to(opt.device)

        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)






        def evaluate():
            print("  Number of testing samples %d" % len(test_ind))
            print('  Start testing...')
            model.load_state_dict(torch.load(fold_model_path))
            model.eval()
            node_logits, _ = model(features_cuda, edge_index, edgenet_input)
            # save(logits_test, test_ind, fold, y[test_ind])
            logits_test = node_logits[test_ind].detach().cpu().numpy()
            # save(logits_test, test_ind1, fold, x[test_ind1])
            corrects[fold], accs[fold],spe[fold], sen[fold]= accuracy(logits_test, y[test_ind])
            aucs[fold] = auc(logits_test, y[test_ind])
            prfs[fold] = prf(logits_test, y[test_ind])

            print("  Fold {} test accuracy {:.5f}, AUC {:.5f}, Specificity {:.5f}, Specificity {:.5f}".format(fold, accs[fold], aucs[fold],spe[fold],sen[fold]))


        evaluate()

    print("\r\n========================== Finish ==========================")
    n_samples = disease.shape[0]
    acc_nfold = np.sum(corrects) / n_samples
    print("=> Average test accuracy in {}-fold CV: {:.5f}".format(n_folds, acc_nfold))
    print("=> Average test AUC in {}-fold CV: {:.5f}".format(n_folds, np.mean(aucs)))
    se, sp, f1 = np.mean(prfs, axis=0)
    print("=> Average test sensitivity {:.4f}, specificity {:.4f}, F1-score {:.4f}".format(se, sp, f1))
    # auc, se, sp, ka = auc_la()
    # print("=> ka {:.4f} ,auc {:.4f}, sensitivity {:.4f}, specificity {:.4f}".format(ka, auc, se, sp))
