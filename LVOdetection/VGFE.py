import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

'''
[ -5  -8   0   8   5 ]
[ -10 -16  0  16  10 ]
[ -20 -32  0  32  20 ]
[ -10 -16  0  16  10 ]
[ -5  -8   0   8   5 ]

[ -5  -10 -20 -10 -5 ]
[ -8  -16 -32 -16 -8 ]
[  0   0   0   0   0 ]
[  8   16  32  16  8 ]
[  5   10  20  10  5 ]
'''


def get_sobel(in_chan, out_chan, stride=1):
    '''
    filter_x = np.array([
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3],
    ]).astype(np.float32)
    filter_y = np.array([
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3],
    ]).astype(np.float32)
    '''
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    # 固定参数，防止学习过程中参数变化
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=False, groups=out_chan)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=False, groups=out_chan)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    # 返回参数固定的两个卷积核
    return sobel_x, sobel_y


# the edge of whole image
class all_edge(nn.Module):
    def __init__(self):
        super(all_edge, self).__init__()
        self.sobel_x, self.sobel_y = get_sobel(1, 1)

    def forward(self, x):
        sobel_x = self.sobel_x(x)
        sobel_y = self.sobel_y(x)
        edge = torch.sqrt(torch.pow(sobel_x, 2) + torch.pow(sobel_y, 2))
        return edge


# the edge of background and foreground
class roi_edge(nn.Module):
    def __init__(self):
        super(roi_edge, self).__init__()
        self.sobel_x, self.sobel_y = get_sobel(1, 1)

    def forward(self, x):
        roi_mask = torch.where(x < 0, 0.0, 1.0)
        sobel_x = self.sobel_x(roi_mask)
        sobel_y = self.sobel_y(roi_mask)
        edge = torch.sqrt(torch.pow(sobel_x, 2) + torch.pow(sobel_y, 2))
        edge = torch.where(edge != 0, 1.0, 0.0)
        return edge

# 特征层的VEM
class EAM(nn.Module):
    def __init__(self, chan, kernel_size=5):
        super(EAM, self).__init__()
        self.chan = chan
        self.sobel_x, self.sobel_y = get_sobel(chan, chan)
        self.bn = nn.BatchNorm2d(chan, eps=1e-5, momentum=0.01, affine=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x_sobel_x = self.sobel_x(x)
        x_sobel_y = self.sobel_y(x)
        x_sobel = torch.sqrt(torch.pow(x_sobel_x, 2) + torch.pow(x_sobel_y, 2))
        x_sobel = self.bn(x_sobel)

        #x_sobel = self.maxpool(x_sobel)
        #x_sobel = torch.sigmoid(x_sobel)

        return x_sobel

# 提取输入图像的血管粗粒度信息
# 仅对边缘信息进行膨胀和腐蚀
class VAM_v1(nn.Module):
    def __init__(self, stride=1, kernel_size=3):
        super(VAM_v1, self).__init__()
        self.all_edge = all_edge()
        self.roi_edge = roi_edge()
        self.dilate = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        # self.gap_filling = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.erode = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.norm = nn.BatchNorm2d(1)

    def forward(self, x):
        a_edge = self.all_edge(x)
        r_edge = self.roi_edge(x)
        # 是否二值化求mask
        edge = (1.0 - r_edge) * a_edge
        # res = edge
        # mean = edge.mean()
        # std = edge.std()
        # edge = torch.where(edge > mean+1*std, edge, 0)

        # dilate ,gap filling
        # 最大值池化
        edge = self.dilate(edge)
        # edge = self.gap_filling(edge)
        # 对edge求负后最大值池化模拟腐蚀的作用
        # f
        edge = -self.erode(-edge)

        # print(mean, std)
        edge = self.norm(edge)
        # mean = edge.mean()
        # std = edge.std()
        # print(mean, std)

        return edge


# 粗粒度血管感知模块 Vessel Aware Module
# 阈值筛选并二值化后进行膨胀腐蚀操作
class VAM_v2(nn.Module):
    def __init__(self, stride=1, kernel_size=3, threshold=3):
        super(VAM_v2, self).__init__()
        self.all_edge = all_edge()
        self.roi_edge = roi_edge()
        self.threshold = threshold
        # 使用maxpooling模拟膨胀和腐蚀
        self.dilate = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        self.erode = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        a_edge = self.all_edge(x)
        r_edge = self.roi_edge(x)
        # 是否二值化求mask
        edge = (1.0 - r_edge) * a_edge
        # res = edge

        # 利用阈值获取二值化mask
        mean = edge.mean()
        std = edge.std()
        # 阈值guolv
        edge = torch.where(edge > mean + 2.0 * std, 1.0, 0.0)

        # dilate ,gap filling
        # 最大值池化
        edge = self.dilate(edge)
        # 对edge求负后最大值池化模拟腐蚀的作用
        edge = -self.erode(-edge)
        return edge * x


from timm.models.layers import SqueezeExcite


class VGFE(nn.Module):
    def __init__(self,channel,kernel_size=3):
        super(VGFE, self).__init__()

        self.channel_attn = SqueezeExcite(channels=channel, rd_ratio=0.25)

        self.conv = nn.Conv2d(in_channels=3, out_channels=1,
                              kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(1)
        #self.bn2 = nn.BatchNorm2d(channel)
        #self.relu = nn.ReLU()

    def forward(self, x, vess):
        res = self.channel_attn(x)
        maxpool = res.argmax(dim=1, keepdim=True)
        avgpool = res.mean(dim=1, keepdim=True)
        spatial_attn = torch.cat([maxpool, avgpool, vess], dim=1)
        spatial_attn = self.conv(spatial_attn)
        spatial_attn = self.bn(spatial_attn)
        spatial_attn = torch.sigmoid(spatial_attn)
        res = res * spatial_attn
        x = x + res
        return x

# without the guide of coarse-garined vessel feature
class HybridAttention(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super(HybridAttention, self).__init__()
        self.channel_attn = SqueezeExcite(channels=channel, rd_ratio=0.25)
        self.conv = nn.Conv2d(in_channels=2, out_channels=1,
                              kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        res = self.channel_attn(x)

        maxpool = res.argmax(dim=1, keepdim=True)
        avgpool = res.mean(dim=1, keepdim=True)
        spatial_attn = torch.cat([maxpool, avgpool], dim=1)
        spatial_attn = self.conv(spatial_attn)
        spatial_attn = self.bn(spatial_attn)
        spatial_attn = torch.sigmoid(spatial_attn)
        res = res * spatial_attn
        x = x + res
        return x