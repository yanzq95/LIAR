import torch
import torch.nn as nn
from .sample_tools import DCN_layer_illu
import torch.nn.functional as F
from mmdet.models import NECKS


# Block
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



@NECKS.register_module()
class IGS(nn.Module):
    def __init__(self, channels_in, channels_out, scale_factor=0.5,align_corners=True,kernel_size=3):
        super(IGS, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.DA_layer = DCN_layer_illu(channels_in, channels_out, kernel_size,scale_factor,align_corners)

        # [B*N,1,H,W] -> [B*N,C,fH,fW] 对光照图下采样
        self.res_block_1 = ResBlock(1, channels_out//2,stride=2)
        self.res_block_2 = ResBlock(channels_out//2, channels_out//2,stride=2)
        self.res_block_3 = ResBlock(channels_out//2, channels_out//2,stride=2)
        self.res_block_4 = ResBlock(channels_out//2, channels_out,stride=2)

    def forward(self, x, illu_map):
        """
        Args:
            x: [B,N,C,fH,fW]
            enhance_ouput['ilist'][0]: (B*N,C,H,W) 光照图原图的shape
        Returns:
            out: [B,N,C,fH,fW]
        """
        # 对 x reshape (B,N,C,fH,fW) -> (B*N,C,fH,fW)
        if len(x.shape) == 5:
            B, N, C, H, W = x.shape
            x_in = x.view(B*N, C, H, W).contiguous()
        elif len(x.shape) == 4:
            x_in = x

        # 光照图
        illu_map_d = self.res_block_1(illu_map)
        illu_map_d = self.res_block_2(illu_map_d)
        illu_map_d = self.res_block_3(illu_map_d)
        illu_map_d = self.res_block_4(illu_map_d) # (B,C,fH,fW)

        # DCN
        x_out = self.DA_layer(x_in, illu_map_d,illu_map)

        out = x_out + x_in

        if len(x.shape) == 5:
            out_return = out.view(B, N, C, H, W)
        elif len(x.shape) == 4:
            out_return = out

        return out_return

