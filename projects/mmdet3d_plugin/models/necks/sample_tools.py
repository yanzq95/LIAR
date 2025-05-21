import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from mmcv.ops import modulated_deform_conv2d
import os
CNT = 0

class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_feat)
        )

    def forward(self, x):
        return nn.LeakyReLU(0.1, True)(self.backbone(x) + self.shortcut(x))
        


class DCN_layer_illu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor=0.5,align_corners=True,stride=1, padding=1, dilation=1,
                 groups=1, deformable_groups=1, bias=True, extra_offset_mask=True):
        super(DCN_layer_illu, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))

        self.extra_offset_mask = extra_offset_mask
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding),
            bias=True
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.init_offset()
        self.reset_parameters()
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input_feat, inter, illu_map):
        """
        :param input_feat: [B, C, H, W]
        :param inter: [B, C, H, W] 用于生成 offset和 mask
        :return: [B, C, H, W]
        """
        # import pdb;pdb.set_trace()
        out = self.conv_offset_mask(inter) # [B, 3*卷积核size, H, W]  3: xy偏置+mask
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1) # (6, 18, 16, 44) (B, 2*卷积核size^2, H, W) 每个像素上，都有kernel_size^2个采样点，每个采样点有2个偏移量
        mask = torch.sigmoid(mask) # (6, 9, 16, 44)
        if self.align_corners:
            illu_map_rescale = F.interpolate(illu_map,(16,44),mode='bilinear',align_corners=True)
        else:
            illu_map_rescale = F.interpolate(illu_map,(16,44),mode='bilinear',align_corners=False)
        eps = 1e-6  # 防止除零
        illum_inv = 1.0 / (illu_map_rescale + eps)  # 光照越小，值越大
        illum_inv_norm = (illum_inv - illum_inv.min()) / (illum_inv.max() - illum_inv.min()) # (B,1,16,44)
        scale = 1.0 + self.scale_factor*illum_inv_norm
        offset = offset*scale
        
        
        # global CNT
        # import numpy as np
        # save_path = os.path.join(r"/opt/data/private/test/ideals/nightocc/basemodel_clean/vis_dcn_offsets/offsets_stereo",str(CNT)+'.npz')
        
        # np.savez(
        #     save_path,
        #     offset=offset.cpu().numpy(),
        #     mask=mask.cpu().numpy(),
        #     illu_map=illu_map_rescale.cpu().numpy()
        # )
        # CNT = CNT + 1
        # # print(save_path)

        return modulated_deform_conv2d(input_feat.contiguous(), offset, mask, self.weight, self.bias, self.stride,
                                       self.padding, self.dilation, self.groups, self.deformable_groups)

