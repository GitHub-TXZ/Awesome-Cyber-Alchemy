import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 常见的图像差异计算方法
LossFunc = [
    'L1',  
    'MSE',  
    'ED',  
    'Cosine',  
]


# 约束对称性和不对称性的损失函数
# special for LVO task, classification


# 约束对称性和不对称性的损失函数
class SymLoss(nn.Module):
    def __init__(self,mod='MSE',weight=None):
        """
        初始化自定义损失函数。
        """
        super(SymLoss, self).__init__()

        self.weight = weight
        self.mod = mod

        # 在这里进行初始化操作，可以包括损失函数的参数设置等

    def forward(self, feature_map, label):
        """
        计算损失函数的前向传播。
        feature_map: b c h w
        label: b x 1
        """
        global bx1
        flip = torch.flip(feature_map, dims=[3])
        if self.mod=='MSE':
            # MSE改编而来
            SE = (feature_map - flip) ** 2
            # bx1 = torch.sum(SE, dim=(1, 2, 3))
            bx1 = torch.mean(SE, dim=(1, 2, 3))
        elif self.mod=='L1':
            # L1
            L1 = torch.abs(feature_map - flip)
            # bx1 = torch.sum(SE, dim=(1, 2, 3))
            bx1 = torch.mean(L1, dim=(1, 2, 3))
        if self.mod=='ED':
            sqr_diff = (feature_map - flip) ** 2
            sum_sqr_diff = torch.sum(sqr_diff,dim=(1,2,3))
            #distance
            bx1 = torch.sqrt(sum_sqr_diff)

        weights = torch.where(label == 1, 1.0, 1.5)


        #标签不对称的
        #total_loss = torch.where(label == 1, 1/bx1, bx1)
        total_loss = torch.where(label == 1, torch.exp(-bx1), 1.0 - torch.exp(-bx1))
        total_loss = total_loss * weights
        #total_loss = torch.where((label == 1) , -bx1, bx1)
        '''
        #根据label保留对称数据
        sym = torch.where(label == 0, torch.tensor(1.), 0.)
        #根据label保留不对称数据
        dis_sym = torch.where(label == 1, torch.tensor(-1.), 0.)
        #对于对称的数据，差异越大，数值越大
        sym_loss = sym * bx1
        dis_sym_loss = dis_sym * bx1
        #设置阈值，防治数值过大
        #对于不对称的数据，差异越大，数值越小
        #dis_sym_loss = torch.where(dis_sym==1., 1.0/dis_sym_loss, 0.)
        total = sym_loss + dis_sym_loss
        return torch.mean(total)
        '''
        #print(total_loss)
        return torch.mean(total_loss)

class SymCosineLoss(nn.Module):
    def __init__(self, weight = None):
        super(SymCosineLoss, self).__init__()
        self.weight = weight

    def forward(self, x, label):
        # x (B C H W)
        # label (B)
        x = F.normalize(input=x, p=2, dim=3)
        x_flip = x.flip(dims=[3])
        cos_sim = F.cosine_similarity(x, x_flip, dim=3)
        cos_sim = torch.mean(cos_sim, dim=[1,2])
        #cos_sim =
        # 如果对称 相似度高 值接近1
        sym = torch.where(label == 0, torch.exp(-cos_sim), 0)
        # 如果不对称 相似度低 值接近0
        dis_sym = torch.where(label == 1, torch.exp(cos_sim), 0)
        loss = sym + dis_sym
        #loss = loss * self.weight
        return torch.mean(loss)

#多层对称性监督的损失函数
class MultiLayerSymLoss(nn.Module):
    def __init__(self, mod='MSE', weight=None):
        super().__init__()
        self.loss_func = SymLoss(weight=weight,mod=mod)

    def forward(self, feature_maps, label):
        #构建一个初始值为0的tensor
        #loss = torch.tensor(0.0)
        loss = 0
        for feature_map in feature_maps:
            loss += self.loss_func(feature_map, label)
        return loss

if __name__ == '__main__':
    #构建一个随机tensor
    dis_sym = np.random.rand(1, 1, 8, 8)
    symmetric_tensor = np.random.rand(1, 1, 8, 8)  # 生成随机张量
    symmetric_tensor = 0.5 * (symmetric_tensor + np.flip(symmetric_tensor, axis=3))  # 水平对称
    bat2 = np.concatenate((symmetric_tensor, dis_sym), axis=0)
    feature_maps = [torch.from_numpy(bat2)]

    fn = MultiLayerSymLoss()
    loss = fn(feature_maps, torch.tensor([1,0]))
    loss = loss.item()
    print(loss)
