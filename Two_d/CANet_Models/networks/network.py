import torch
import torch.nn as nn
import torch.nn.functional as F

from CANet_Models.layers.modules import conv_block, UpCat, UpCatconv, UnetDsv3, UnetGridGatingSignal3
from CANet_Models.layers.grid_attention_layer import GridAttentionBlock2D, MultiAttentionBlock
from CANet_Models.layers.channel_attention_layer import SE_Conv_Block
from CANet_Models.layers.scale_attention_layer import scale_atten_convblock
from CANet_Models.layers.nonlocal_layer import NONLocalBlock2D
import argparse

class Comprehensive_Atten_Unet(nn.Module):
    def __init__(self,in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1),device='cuda:1',out_size=(224,300),image_size=224):
        super(Comprehensive_Atten_Unet, self).__init__()

        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_size = out_size
        self.image_size = image_size

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # attention blocks
        # self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[1],
        #                                             inter_channels=filters[0])
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[4], inter_channels=filters[4] // 4)

        # upsampling
        self.up_concat4 = UpCat(filters[4], filters[3], self.is_deconv,device=device)
        self.up_concat3 = UpCat(filters[3], filters[2], self.is_deconv,device=device)
        self.up_concat2 = UpCat(filters[2], filters[1], self.is_deconv,device=device)
        self.up_concat1 = UpCat(filters[1], filters[0], self.is_deconv,device=device)
        self.up4 = SE_Conv_Block(filters[4], filters[3], drop_out=True)
        self.up3 = SE_Conv_Block(filters[3], filters[2])
        self.up2 = SE_Conv_Block(filters[2], filters[1])
        self.up1 = SE_Conv_Block(filters[1], filters[0])

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=4, scale_factor=self.out_size)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=4, scale_factor=self.out_size)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=4, scale_factor=self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=4, kernel_size=1)

        self.scale_att = scale_atten_convblock(in_size=16, out_size=4)
        # final conv (without any concat)
        self.final = nn.Sequential(nn.Conv2d(4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        # 在这里上采样至 (224, 300)
        inputs = F.interpolate(inputs, size=(224, 300), mode='bilinear', align_corners=True)
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        up4 = self.up_concat4(conv4, center)
        g_conv4 = self.nonlocal4_2(up4)

        up4, att_weight4 = self.up4(g_conv4)
        g_conv3, att3 = self.attentionblock3(conv3, up4)

        # atten3_map = att3.cpu().detach().numpy().astype(np.float)
        # atten3_map = ndimage.interpolation.zoom(atten3_map, [1.0, 1.0, 224 / atten3_map.shape[2],
        #                                                      300 / atten3_map.shape[3]], order=0)

        up3 = self.up_concat3(g_conv3, up4)
        up3, att_weight3 = self.up3(up3)
        g_conv2, att2 = self.attentionblock2(conv2, up3)

        # atten2_map = att2.cpu().detach().numpy().astype(np.float)
        # atten2_map = ndimage.interpolation.zoom(atten2_map, [1.0, 1.0, 224 / atten2_map.shape[2],
        #                                                      300 / atten2_map.shape[3]], order=0)

        up2 = self.up_concat2(g_conv2, up3)
        up2, att_weight2 = self.up2(up2)
        # g_conv1, att1 = self.attentionblock1(conv1, up2)

        # atten1_map = att1.cpu().detach().numpy().astype(np.float)
        # atten1_map = ndimage.interpolation.zoom(atten1_map, [1.0, 1.0, 224 / atten1_map.shape[2],
        #                                                      300 / atten1_map.shape[3]], order=0)
        up1 = self.up_concat1(conv1, up2)
        up1, att_weight1 = self.up1(up1)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        out = self.scale_att(dsv_cat)

        out = self.final(out)
        # 最后将输出恢复到原始图片大小 (image_size)
        out = F.interpolate(out, size=[self.image_size,self.image_size], mode='bilinear', align_corners=True)
        return out

def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive attention network for biomedical Dataset')
    # Model related arguments
    parser.add_argument('--id', default='Comp_Atten_Unet',
                        help='a name for identitying the model. Choose from the following options: Unet')

    # Path related arguments
    parser.add_argument('--root_path', default='./data/ISIC2018_Task1_npy_all',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved_models',
                        help='folder to output checkpoints')

    # optimization related arguments
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 12)')
    parser.add_argument('--lr_rate', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='weights regularizer')
    parser.add_argument('--particular_epoch', default=30, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--save_epochs_steps', default=200, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--resume', default='',
                        help='the checkpoint that resumes from')

    # other arguments
    parser.add_argument('--data', default='ISIC2018', help='choose the dataset')
    parser.add_argument('--out_size', default=(224, 300), help='the output image size')
    parser.add_argument('--val_folder', default='folder0', type=str,
                        help='which cross validation folder')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda:1')
    model = Comprehensive_Atten_Unet(device=device,image_size=224).to(device)
    inputs = torch.randn(1, 3, 224, 224).to(device)
    outputs = model(inputs)
    print(outputs.size())
    print(outputs)