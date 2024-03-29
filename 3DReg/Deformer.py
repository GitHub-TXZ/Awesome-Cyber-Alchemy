"""
Defomer

Original code retrieved from:
https://github.com/CJSOrange/DMR-Deformer

Original paper:
Deformer: Towards Displacement Field Learning for Unsupervised Medical Image Registration

"""
# %%
""" Define the sublayers in Deformer"""
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()


class Local_Attention(nn.Module):
    """Multi-Head local-Attention module"""

    def __init__(self, n_head, n_point, d_model):
        super().__init__()

        self.n_head = n_head
        self.n_point = n_point
        # one linear layer to obtain displacement basis
        self.sampling_offsets = nn.Linear(d_model, n_head * n_point * 3)
        # one linear layer to obtain weight
        self.attention_weights = nn.Linear(2 * d_model, n_head * n_point)
        # self.attention_weights = nn.Linear(2 * d_model, n_head * 3)

    def forward(self, q, k):
        v = torch.cat([q, k], dim=-1)
        n_head, n_point = self.n_head, self.n_point
        sz_b, len_q, len_k = q.size(0), q.size(1), k.size(1)
        # left branch (only moving image)
        sampling_offsets = self.sampling_offsets(q).view(
            sz_b, len_q, n_head, n_point, 3
        )
        # right branch (concat moving and fixed image)
        attn = self.attention_weights(v).view(sz_b, len_q, n_head, n_point, 1)
        # attn = self.attention_weights(v).view(sz_b, len_q, n_head, 3)
        # flow = attn
        # attn = F.softmax(attn, dim=-2)
        # multiple and head-wise average
        flow = torch.matmul(sampling_offsets.transpose(3, 4), attn)
        flow = torch.squeeze(flow, dim=-1)
        # sz_b, len_q, 3
        return torch.mean(flow, dim=-2)


class Deformer_layer(nn.Module):
    """Compose layers"""

    def __init__(self, d_model, n_head, n_point):
        super(Deformer_layer, self).__init__()
        self.slf_attn = Local_Attention(n_head, n_point, d_model)

    def forward(self, enc_input, enc_input1):
        enc_output = self.slf_attn(enc_input, enc_input1)
        return enc_output


class Deformer(nn.Module):
    """
    A encoder model with deformer mechanism.
    :param n_layers: the number of layer.
    :param d_model: the channel of input image [batch,N,d_model].
    :param n_position: input image [batch,N,d_model], n_position=N.
    :param n_head: the number of head.
    :param n_point: the number of displacement base.
    :param src_seq: moving seq [batch,N,d_model]
    :param tgt_seq: fixed seq [batch,N,d_model].
    :return enc_output: sub flow field [batch,N,3].
    """

    def __init__(
        self,
        n_layers,
        d_model,
        n_position,
        n_head,
        n_point,
        dropout=0.1,
        scale_emb=False,
    ):
        super().__init__()

        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [Deformer_layer(d_model, n_head, n_point) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, tgt_seq):
        # -- Forward
        if self.scale_emb:
            src_seq *= self.d_model**0.5
            tgt_seq *= self.d_model**0.5
        enc_output = self.dropout(self.position_enc(src_seq))
        enc_output = self.layer_norm(enc_output)
        enc_output1 = self.dropout(self.position_enc(tgt_seq))
        enc_output1 = self.layer_norm(enc_output1)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, enc_output1)

        return enc_output


# if __name__ == "__main__":
#     x = torch.rand(3, 112 * 96 * 80, 16)
#     y = torch.rand(3, 112 * 96 * 80, 16)
#     b, n, d = x.size()
#     enc = Deformer(n_layers=1, d_model=d, n_position=n, n_head=8, n_point=64)
#     z = enc(x, y)
#     print(z.size())
#     print(torch.min(z))

# %%
import torch.nn as nn
import torch.nn.functional as F
import torch

"""The Refining Network of DMR"""


class Deformable_Skip_Learner(nn.Module):
    def __init__(self, inch):
        super(Deformable_Skip_Learner, self).__init__()

        def make_building_block(
            in_channel, out_channels, kernel_sizes, spt_strides, group=1
        ):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(
                zip(out_channels, kernel_sizes, spt_strides)
            ):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                pad = ksz // 2
                building_block_layers.append(nn.Conv3d(inch, outch, ksz, stride, pad))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 16, 64, 128

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(
            inch[0], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1]
        )
        self.encoder_layer3 = make_building_block(
            inch[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1]
        )
        self.encoder_layer2 = make_building_block(
            inch[2], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1]
        )
        self.encoder_layer1 = make_building_block(
            inch[3], [outch1, outch2, outch3], [5, 5, 5], [1, 1, 1]
        )

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(
            outch3 + 32 * 2, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]
        )
        self.encoder_layer3to2 = make_building_block(
            outch3 + 32 * 2, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]
        )
        self.encoder_layer2to1 = make_building_block(
            outch3 + 16 * 2, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]
        )

        # Decoder layers
        self.decoder1 = nn.Sequential(
            nn.Conv3d(outch3, outch3, (3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.ReLU(),
            nn.Conv3d(outch3, outch2, (3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.ReLU(),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv3d(outch2, outch2, (3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.ReLU(),
            nn.Conv3d(outch2, outch1, (3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.ReLU(),
        )

        self.decoder3 = nn.Sequential(
            nn.Conv3d(outch1, outch1, (3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.ReLU(),
            nn.Conv3d(outch1, 3, (3, 3, 3), padding=(1, 1, 1), bias=True),
        )

    def interpolate_dims(self, hypercorr, spatial_size=None):
        bsz, ch, d, w, h = hypercorr.size()
        hypercorr = F.interpolate(
            hypercorr, (2 * d, 2 * w, 2 * h), mode="trilinear", align_corners=True
        )
        return hypercorr

    def forward(self, hypercorr_pyramid, moving_feat, fixed_feat):
        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])
        hypercorr_sqz1 = self.encoder_layer1(hypercorr_pyramid[3])

        # Propagate encoded 3D-tensor (Mixing building blocks)
        hypercorr_sqz4 = self.interpolate_dims(
            hypercorr_sqz4, hypercorr_sqz3.size()[-6:-3]
        )
        hypercorr_mix43 = 2 * hypercorr_sqz4 + hypercorr_sqz3  # add
        hypercorr_mix43 = torch.cat(
            [hypercorr_mix43, moving_feat[-2], fixed_feat[-2]], dim=1
        )  # skip connection
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = self.interpolate_dims(
            hypercorr_mix43, hypercorr_sqz2.size()[-6:-3]
        )
        hypercorr_mix432 = 2 * hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = torch.cat(
            [hypercorr_mix432, moving_feat[-3], fixed_feat[-3]], dim=1
        )
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        hypercorr_mix432 = self.interpolate_dims(
            hypercorr_mix432, hypercorr_sqz1.size()[-6:-3]
        )
        hypercorr_mix4321 = 2 * hypercorr_mix432 + hypercorr_sqz1
        hypercorr_mix4321 = torch.cat(
            [hypercorr_mix4321, moving_feat[-4], fixed_feat[-4]], dim=1
        )
        hypercorr_mix4321 = self.encoder_layer2to1(hypercorr_mix4321)

        # Decode the encoded 3D-tensor
        hypercorr_decoded = self.decoder1(hypercorr_mix4321)
        upsample_size = (
            hypercorr_decoded.size(-3) * 2,
            hypercorr_decoded.size(-2) * 2,
            hypercorr_decoded.size(-1) * 2,
        )
        hypercorr_decoded = 2 * F.interpolate(
            hypercorr_decoded, upsample_size, mode="trilinear", align_corners=True
        )
        hypercorr_decoded = self.decoder2(hypercorr_decoded)
        logit_mask = self.decoder3(hypercorr_decoded)

        return logit_mask


# if __name__ == "__main__":
#     import torch

#     corr = []
#     corr.append(torch.rand(2, 3, 14, 12, 10))
#     corr.append(torch.rand(2, 3, 28, 24, 20))
#     corr.append(torch.rand(2, 3, 56, 48, 40))
#     corr.append(torch.rand(2, 3, 112, 96, 80))
#     hpn_learner = Deformable_Skip_Learner([3, 3, 3, 3])
#     moving_feat = []
#     moving_feat.append(torch.rand(2, 16, 112, 96, 80))
#     moving_feat.append(torch.rand(2, 32, 56, 48, 40))
#     moving_feat.append(torch.rand(2, 32, 28, 24, 20))
#     moving_feat.append(torch.rand(2, 64, 14, 12, 10))

#     fixed_feat = []
#     fixed_feat.append(torch.rand(2, 16, 112, 96, 80))
#     fixed_feat.append(torch.rand(2, 32, 56, 48, 40))
#     fixed_feat.append(torch.rand(2, 32, 28, 24, 20))
#     fixed_feat.append(torch.rand(2, 64, 14, 12, 10))

#     y = hpn_learner(corr, moving_feat, fixed_feat)
#     print(y.shape)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# import configs as configs
# from VIT import ViTVNet
# from Resnet3D import generate_model

"""
Encoder of DMR, 4 3D conv layer for 1/2, 1/4, 1/8, 1/16 scale
    :param dim: the dimension of input [batch,dim,d,w,h].
    :param bn: whether use batch normalization or not, True->use.
    :param x: the input medical image [batch,dim,d,w,h].
    :return x: feature at 1/16 scale.
    :return x_enc: feature at 1/2, 1/4, 1/8, 1/16 scale.
"""


class Encoder(nn.Module):
    def __init__(self, dim, bn=True):
        super(Encoder, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = [16, 32, 32, 64]
        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(self.enc_nf)):
            prev_nf = 1 if i == 0 else self.enc_nf[i - 1]
            self.enc.append(
                conv_block(dim, prev_nf, self.enc_nf[i], 4, 2, batchnorm=bn)
            )

    def forward(self, x):
        # Get encoder activations
        x_enc = [x]
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            x_enc.append(x)
        return x, x_enc


def conv_block(
    dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False
):
    conv_fn = getattr(nn, "Conv{0}d".format(dim))
    bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
    if batchnorm:
        layer = nn.Sequential(
            conv_fn(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            ),
            bn_fn(out_channels),
            nn.LeakyReLU(0.2),
        )
    else:
        layer = nn.Sequential(
            conv_fn(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            ),
            nn.LeakyReLU(0.2),
        )
    return layer


"""
Implementation details of DMR:
    :param dim: the dimension of input [batch,dim,d,w,h].
    :param vol: the size of medical image in LPBA40 [160,192,160] or OASIS [224,192,160]
    :param layer: From coarse to fine, decide which layers use Deformer. Otherwise use concatenation.
    if layer=2, means 1/16 and 1/8 resolution use Deformer. 1/4 and 1/2 use concatenation.
    :return flow: displacement field at full resolution [batch,3,d,w,h].
    :return corrs: sub displacement field at 1/2,1/4,1/8,1/16 resolution.
"""


class DMR(nn.Module):
    def __init__(self, dim, vol, layer):
        super(DMR, self).__init__()
        # One conv to get the flow field
        self.backbone = Encoder(dim)
        self.derlearn = Deformable_Skip_Learner([3, 3, 3, 3])
        self.derlayer = nn.ModuleList()
        self.vol = vol
        self.layer = layer
        d, w, h = vol
        d = d // 32
        w = w // 32
        h = h // 32
        # Deformer
        self.derlayer.append(
            Deformer(
                n_layers=1, d_model=64, n_position=d * w * h * 8, n_head=8, n_point=64
            )
        )
        self.derlayer.append(
            Deformer(
                n_layers=1, d_model=32, n_position=d * w * h * 64, n_head=8, n_point=64
            )
        )
        self.derlayer.append(
            Deformer(
                n_layers=1, d_model=32, n_position=d * w * h * 512, n_head=8, n_point=64
            )
        )
        self.derlayer.append(
            Deformer(
                n_layers=1,
                d_model=16,
                n_position=d * w * h * 4096,
                n_head=8,
                n_point=64,
            )
        )
        # Deformer->VIT
        """
        self.config_vit = configs.get_3DReg_config()
        self.derlayer.append(
            ViTVNet(self.config_vit, in_channels=64, img_size=(2*d, 2*w, 2*h)))
        self.derlayer.append(
            ViTVNet(self.config_vit, in_channels=32, img_size=(4*d, 4*w, 4*h)))
        self.derlayer.append(
            ViTVNet(self.config_vit, in_channels=32, img_size=(8*d, 8*w, 8*h)))
        self.derlayer.append(
            ViTVNet(self.config_vit, in_channels=16, img_size=(16*d, 16*w, 16*h)))
        """
        # Deformer->3D ResNet
        """
        self.derlayer.append(
            generate_model(50, n_input_channels=64*2))
        self.derlayer.append(
            generate_model(50, n_input_channels=32*2))
        self.derlayer.append(
            generate_model(50, n_input_channels=32*2))
        self.derlayer.append(
            generate_model(50, n_input_channels=16*2))
        """

    def forward(self, moving, fixed):
        input = torch.cat([moving, fixed], dim=0)
        _, feature = self.backbone(input)
        b = moving.shape[0]
        moving_feature = []
        fixed_feature = []
        corrs = []

        for i in feature:
            moving_feature.append(i[0:b])
            fixed_feature.append(i[b:])

        for i in range(1, self.layer + 1):
            b, c, d, w, h = moving_feature[-i].size()
            moving_feat = moving_feature[-i].clone().flatten(2).transpose(1, 2)
            fixed_feat = fixed_feature[-i].clone().flatten(2).transpose(1, 2)
            corr = self.derlayer[i - 1](moving_feat, fixed_feat)
            corr = corr.transpose(1, 2).view(b, 3, d, w, h)
            corrs.append(corr)

        for i in range(self.layer + 1, 5):
            corrs.append(torch.cat([moving_feature[-i], fixed_feature[-i]], dim=1))

        flow = self.derlearn(corrs, moving_feature, fixed_feature)
        return flow  # , corrs


"""Refer to STN (paper "Spatial Transformer Networks")"""


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode="bilinear"):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer("grid", grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=False)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cuda0 = torch.device("cuda:0")
    x = torch.rand(1, 1, 96, 96, 96)
    y = torch.rand(1, 1, 96, 96, 96)
    dim = 3

    vol = [96, 96, 96]
    model = DMR(dim, vol, layer=4)
    # result = get_parameter_number(model)
    # print(result["Total"], result["Trainable"])  # 打印参数量
    # from thop import profile

    # flops, params = profile(
    #     model,
    #     (
    #         x,
    #         y,
    #     ),
    # )
    # print("flops: ", flops, "params: ", params)
    flow, corrs = model(x, y)

    stn = SpatialTransformer(vol)
    moved = stn(x, flow)
    print(moved.size())
