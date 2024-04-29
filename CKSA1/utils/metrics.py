from math import log10
import torch
import numpy as np 
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import torch.nn.functional as F
from scipy.special import softmax
import scipy.stats
from sklearn.metrics import roc_auc_score
from medpy.metric.binary import sensitivity
from medpy.metric.binary import specificity
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import cohen_kappa_score,confusion_matrix
import pingouin as pg
from scipy.stats import t
a = 0
def PSNR(mse, peak=1.):
	return 10 * log10((peak ** 2) / mse)

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(preds, labels):
    """Accuracy, auc with masking.Acc of the masked samples"""
    x = np.sum(labels)
    y = len(labels) - x
    if a == 0:
        pred = preds
    else:
        pred = np.argmax(preds, 1)
    #pred = preds
    spe = specificity(pred, labels)
    sen = sensitivity(pred, labels)
    correct_prediction = np.equal(pred, labels).astype(np.float32)
    sen_sum = round(x * sen)
    spe_sum = round(y * spe)
    return np.sum(correct_prediction), np.mean(correct_prediction),spe,sen,spe_sum,sen_sum
def sa(sub_ids,preds,fold,n):
    base_path = r"/home/lining/Data/Res/"
    save_path = os.path.join(base_path, f"I_{fold}.csv")
    A = np.argmax(preds, 1)
    pos_probs = softmax(preds, axis=1)
    info = {"TEST_ID": [],
            "pred": [],
            "prob_0": [],
            "prob_1": []
            }
    for i in range(len(preds)):
        info["TEST_ID"].append(sub_ids[n - 100 + i])
        info["pred"].append(A[i])
        info["prob_0"].append(pos_probs[i][0])
        info["prob_1"].append(pos_probs[i][1])
    df = pd.DataFrame(info)
    df.to_csv(save_path)

def compute_confidence_interval(data):
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    z = 1.96  # 95% confidence interval (for large sample sizes)
    lower_bound = mean - z * (std / np.sqrt(n))
    upper_bound = mean + z * (std / np.sqrt(n))
    return lower_bound, upper_bound
import os
import pandas as pd
import glob
import csv
def save(preds, labels, local,named,region,asdw):
    preds = np.argmax(preds, 1)
    base_path = r"/homeb/lining/Data/experiment/KSR"
    save_path = os.path.join(base_path, named)
    if not os.path.exists(save_path):
        # 如果路径不存在，创建目录
        os.makedirs(save_path)
    save_path = os.path.join(save_path, str(asdw))
    if not os.path.exists(save_path):
        # 如果路径不存在，创建目录
        os.makedirs(save_path)
    save_path_csv = os.path.join(save_path, f"{region}.csv")
    with open(save_path_csv, mode="a", newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            header = [f"ids", f"{region}_true", f"{region}_pred"]
            writer.writerow(header)
        for i in range(len(preds)):
            row = [local[i], labels[i], preds[i]]
            writer.writerow(row)

def Over_all(named, asdw):
    base_path = r"/homeb/lining/Data/experiment/KSR"
    save_path = os.path.join(base_path, named)
    save_path = os.path.join(save_path, str(asdw))
    save_path1 = os.path.join(save_path, f"{named}_All")
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
    csv_files = glob.glob(os.path.join(save_path, '*.csv'))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        column_to = 'ids'
        df = df.sort_values(by=column_to, ascending=True)
        df = df.drop_duplicates(subset=column_to, keep='first')
        df.to_csv(csv_file, index=False)
    combined_data = pd.DataFrame()
    column_name = ["ids"]
    for csv_file in csv_files:
        base = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)
        column_name.append(f"{base[:-4]}_true")
        column_name.append(f"{base[:-4]}_pred")
        if not combined_data.empty:
            df = df.iloc[:, 1:]
        combined_data = pd.concat([combined_data,df], axis=1, ignore_index=True)
    save_path_new = os.path.join(save_path1, f"All.csv")
    combined_data.columns = column_name
    combined_data.to_csv(save_path_new, index=False)
    print(f'已合并CSV文件并保存为 {save_path_new}')
    df = pd.read_csv(save_path_new).iloc[:, 1:].values
    a = []
    b = []
    for i in range(0,20,2):
        a.append(i)
    for i in range(1,20,2):
        b.append(i)
    true = df[:,a]
    true1 = np.array(true).flatten()
    true = np.sum(true, axis=1)
    pred = df[:, b]
    pred1 = np.array(pred).flatten()
    pred = np.sum(pred, axis=1)
    print(asdw,"Sdasdasdasd")
    median = np.median(10-true)
    q1 = np.percentile(10-true, 25)
    q3 = np.percentile(10-true, 75)
    print(to_icc(preds=(10-pred), labels=(10-true)))
    pred[pred < 6] = 0
    pred[pred >= 6] = 1
    true[true < 6] = 0
    true[true >= 6] = 1
    print("4为界限:", accuracy(pred,true))
    print("All_region:", accuracy(pred1, true1))
    print(auc(pred, true))
    print()


def auc_la():
    base_path = r"/home/lining/GCN/M3/CKSA/data/Result"
    save_path = os.path.join(base_path, f"All_I.csv")
    if not os.path.exists(save_path):
        return 0,0,0
    df = pd.read_csv(save_path)
    auc_out = roc_auc_score(df["TRUE"], df["prob_1"])
    se = sensitivity(df["pred"], df["TRUE"])
    sp = specificity(df["pred"], df["TRUE"])
    ka = cohen_kappa_score(df["pred"], df["TRUE"])
    os.remove(save_path)
    return auc_out,se,sp,ka



def auc(preds, labels, is_logit=True):
    ''' input: logits, labels  '''
    if a == 0:
        pred = preds
    else:
        pred = np.argmax(preds, 1)
    try:
        auc_out = roc_auc_score(labels, pred)
    except:
        auc_out = 0
    return auc_out
def prf(preds, labels, is_logit=True):
    ''' input: logits, labels  '''
    if a == 0:
        pred = preds
    else:
        pred = np.argmax(preds, 1)
    #pred_lab = preds
    p,r,f,s  = precision_recall_fscore_support(labels, pred, average='binary')
    return [p,r,f]

def interval1_std(data):
    mean = np.mean(data)
    #return np.std(data)
    std_data = np.std(data) / np.sqrt(len(data))
    confidence = 0.95
    n = len(data)
    dof = n - 1
    interval = t.interval(confidence, dof, loc=mean, scale=std_data)
    radius = (interval[1] - interval[0]) / 2
    return radius

def kappa(preds,GT):
    if a == 0:
        pred = preds
    else:
        pred = np.argmax(preds, 1)
    #pred = preds
    ka = cohen_kappa_score(pred,GT)
    return ka


def to_icc(preds, labels):
    if a == 0:
        pred = preds
    else:
        pred = np.argmax(preds, 1)
    matrix = confusion_matrix(labels,pred)
    pred_list = []
    real_list = []
    # 方阵
    leng = len(matrix)
    # cnt=0
    for i in range(leng):
        for j in range(leng):
            value = matrix[i][j]
            pred_list.extend([j for k in range(value)])
            real_list.extend([i for k in range(value)])

    icc = icc_caculate(pred_list, real_list)
    icc1k = icc["ICC"]
    icc = icc1k[5]
    return icc


def icc_caculate(pred_list, real_list):
    id_list = [i for i in range(len(pred_list))]
    id_list.extend([i for i in range(len(real_list))])
    judge = ['pre' for i in range(len(pred_list))]
    judge.extend('real' for i in range(len(pred_list)))
    score_list = pred_list
    score_list.extend(real_list)
    dic = {"id": id_list, "judge": judge, "score": score_list}
    excel = pd.DataFrame(dic)
    icc = pg.intraclass_corr(data=excel, targets='id', raters='judge', ratings='score')
    return icc