import torch
import torch.nn as nn
from mmdet.models import NECKS
from torch.nn import Identity
from werkzeug.serving import load_ssl_context
from mmdet3d.models.builder import build_loss


# 学习的是光照 illumination
class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)
        return illu

# 通过逐步消除不理想的光照或噪声成分来优化输入图像，使得每个阶段的输入更接近真实的反射分量，从而帮助后续增强网络获得更好的低光图像增强效果。
class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = input - fea

        return delta


@NECKS.register_module()
class Light_Enhance_V3(nn.Module):
    def __init__(self, stage=3,enhance_layer=3,enhance_channel=1,calibrate_layer=3,calibrate_channel=1,loss_light=None):
        super(Light_Enhance_V3, self).__init__()
        self.stage = stage
        self.enhance = EnhanceNetwork(layers=enhance_layer, channels=enhance_channel)
        self.calibrate = CalibrateNetwork(layers=calibrate_layer, channels=calibrate_channel)
        # self.loss_function = build_loss(loss_light)
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input
        for i in range(self.stage):
            inlist.append(input_op)
            i = self.enhance(input_op)
            r = input / i
            r = torch.clamp(r, 0, 1)
            att = self.calibrate(r)
            input_op = input + att
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))
        # ilist：每个stage的光照图
        # rlist：每个stage的反射图
        # inlist：每个stage开始时输入的图像
        # attlist： 每个stage中CalibrateNetwork生成的校正图
        return ilist, rlist, inlist, attlist

    # def get_light_loss(self, output_dict):
    #     inlist, ilist = output_dict['inlist'], output_dict['ilist']
    #     loss = 0
    #     for i in range(self.stage):
    #         loss += self.loss_function(inlist[i], ilist[i])
    #     return loss
