# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, MultiScaleDeformableAttnFunction_fp16

ext_module = ext_loader.load_ext('_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class DA_SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 layer_scale=None,
                 dbound=None,
                 **kwargs
                 ):
        super(DA_SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.dbound = dbound
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        if layer_scale is not None:
            self.layer_scale = nn.Parameter(
                layer_scale * torch.ones(embed_dims),
                requires_grad=True)
        else:
            self.layer_scale = None
        self.init_weight()
        self.count = 0

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                level_start_index=None,
                flag='encoder',
                bev_query_depth=None,
                pred_img_depth=None,
                bev_mask=None,
                per_cam_mask_list=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        # bev_query_depth 表示每个BEV查询点在3D空间中的深度
        N, B, len_query, Z, _ = bev_query_depth.shape # torch.Size([6, 1, 40000, 4, 1])

        # pred_img_depth 表示每个相机的2D深度预测图
        B, N, DC, H, W = pred_img_depth.shape

        bev_query_depth = bev_query_depth.permute(1, 0, 2, 3, 4)  # torch.Size([2, 6, 10000, 4, 1])
        pred_img_depth = pred_img_depth.view(B * N, DC, H, W)
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)  # torch.Size([12, 704, 80])

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)
        indexes = [[] for _ in range(bs)]

        # FBOCC中没有用bev_mask
        if bev_mask is not None:
            per_cam_mask_list_ = per_cam_mask_list & bev_mask[None, :, :, None]
        else:
            per_cam_mask_list_ = per_cam_mask_list
        max_len = 0
        for j in range(bs):
            for i, per_cam_mask in enumerate(per_cam_mask_list_):
                index_query_per_img = per_cam_mask[j].sum(-1).nonzero().squeeze(-1)
                if len(index_query_per_img) == 0:
                    index_query_per_img = per_cam_mask_list[i][j].sum(-1).nonzero().squeeze(-1)[0:1]
                indexes[j].append(index_query_per_img)
                max_len = max(max_len, len(index_query_per_img))


        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])
        bev_query_depth_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 1])


        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[j][i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                bev_query_depth_rebatch[j, i, :len(index_query_per_img)] = bev_query_depth[j, i, index_query_per_img]

                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        bev_query_depth_rebatch = (bev_query_depth_rebatch - self.dbound[0]) / self.dbound[2]
        bev_query_depth_rebatch = torch.clip(torch.floor(bev_query_depth_rebatch), 0, DC - 1).to(torch.long) # torch.Size([1, 6, 8925, 4, 1])

        # #############################################################################
        # tensor_dict = {
        #     'bev_query_depth_rebatch':   bev_query_depth_rebatch,
        #     'indexes':       indexes,
        # }
        # torch.save(tensor_dict, 'bev_query_depth_rebatch.pt')
        # import pdb;pdb.set_trace()
        # #############################################################################

        bev_query_depth_rebatch = F.one_hot(bev_query_depth_rebatch.squeeze(-1),
                                            num_classes=DC)

        # dict(type='DA_MSDeformableAttention')
        queries = self.deformable_attention(query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
                                            key=key, value=value, \
                                            reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len,
                                                                                           D, 2),
                                            spatial_shapes=spatial_shapes, \
                                            level_start_index=level_start_index, \
                                            bev_query_depth=bev_query_depth_rebatch.view(bs * self.num_cams, max_len, D,
                                                                                         DC), \
                                            pred_img_depth=pred_img_depth, \
                                            ).view(bs, self.num_cams, max_len, self.embed_dims)

        for j in range(bs):
            for i in range(num_cams):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        # 聚合多相机结果，对重叠区域取均值
        count = per_cam_mask_list_.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]

        slots = self.output_proj(slots)
        if self.layer_scale is None:
            return self.dropout(slots) + inp_residual
        else:
            return self.dropout(self.layer_scale * slots) + inp_residual


@ATTENTION.register_module()
class DA_MSDeformableAttention(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,    # 特征图数量=1
                 num_points=8,    # 每个注意力头在每个特征图上采样的点数
                 num_Z_anchors=4, # 在 3D 空间中每个 BEV 查询的高度锚点数
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 disable_deformable=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False
        self.disable_deformable = disable_deformable
        self.num_Z_anchors = num_Z_anchors

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)

        self.each_anchor_points = self.num_points // self.num_Z_anchors

        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 1,
            2).repeat(1, self.num_levels, self.each_anchor_points, self.num_Z_anchors, 1)
        for i in range(self.each_anchor_points):
            for j in range(self.num_Z_anchors):
                grid_init[:, :, i, j, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @force_fp32()
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                bev_query_depth=None,
                pred_img_depth=None,

                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        # 生成采样偏移和注意力权重
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        if self.disable_deformable:
            sampling_offsets = sampling_offsets * 0
            attention_weights = attention_weights * 0
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]

            sampling_offsets = sampling_offsets / \
                               offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets # 采样点 = 参考点 + 偏移量
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            depth_reference_points = reference_points.reshape(bs, num_query * num_Z_anchors, 1, 1, 1, 2).contiguous()
            depth_attention_weights = torch.ones_like(depth_reference_points[..., 0]).contiguous()
            # num_query： 8920
            # pred_img_depth.unsqueeze(2).contiguous(): torch.Size([6, 704, 1, 80])
            # depth_reference_points: torch.Size([6, 71360, 1, 1, 1, 2])
            # depth_attention_weights: torch.Size([6, 71360, 1, 1, 1])
            # sampling_locations: (bs ,num_queries, num_heads, num_levels, num_points, 2) 2表示(x, y).

            depth_weights = MultiScaleDeformableAttnFunction.apply(
                pred_img_depth.unsqueeze(2).contiguous(), spatial_shapes[0:1], level_start_index[0:1],
                depth_reference_points,
                depth_attention_weights, self.im2col_step).reshape(bs, num_query, num_Z_anchors, -1)
            # depth_weights: torch.Size([6, 8920, 8, 80])
            # bev_query_depth: torch.Size([6, 8920, 8, 80]) 每个cam每个点的深度分布

            depth_weights = (depth_weights * bev_query_depth).sum(-1)
            depth_weights = depth_weights.unsqueeze(2).repeat(1, 1, num_points, 1).reshape(bs, num_query,
                                                                                           num_all_points)

            # depth_weights: torch.Size([6, 8920, 8])
            # attention_weights: torch.Size([6, 8920, 1, 1, 8])
            attention_weights = attention_weights * depth_weights[:, :, None, None, :]
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return output
