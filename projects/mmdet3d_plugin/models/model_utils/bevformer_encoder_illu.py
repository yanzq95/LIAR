# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE
from distutils.command.build import build

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.models.builder import build_neck
import numpy as np
import torch
import cv2 as cv
import mmcv
import time
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
import torch.nn.functional as F

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class IDP_transformer_encoder(TransformerLayerSequence):
    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 pc_range=None,
                 grid_config=None,
                 data_config=None,
                 return_intermediate=False,
                 down_sampled_illu_map=True,
                 use_gird_sample=False,
                 bev_learning=dict(type='Identity'),
                 fix_bug=False,
                 **kwargs):
        super(IDP_transformer_encoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fix_bug = fix_bug
        self.x_bound = grid_config['x']
        self.y_bound = grid_config['y']
        self.z_bound = grid_config['z']
        self.final_dim = data_config['input_size']
        self.pc_range = pc_range
        self.down_sampled_illu_map = down_sampled_illu_map
        self.fp16_enabled = False
        # bev空间权重学习
        self.use_gird_sample = use_gird_sample
        self.bev_learning = build_neck(bev_learning)

    def get_reference_points(self, H, W, Z=8, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            X = torch.arange(*self.x_bound, dtype=torch.float) + self.x_bound[-1] / 2
            Y = torch.arange(*self.y_bound, dtype=torch.float) + self.y_bound[-1] / 2
            Z = torch.arange(*self.z_bound, dtype=torch.float) + self.z_bound[-1] / 2
            Y, X, Z = torch.meshgrid([Y, X, Z])
            coords = torch.stack([X, Y, Z], dim=-1)
            coords = coords.to(dtype).to(device)
            # frustum = torch.cat([coords, torch.ones_like(coords[...,0:1])], dim=-1) #(x, y, z, 4)

            return coords

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling_with_illu_weight_grid_sample(self, reference_points, cam_params=None, illu_map=None):
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        B, N, _ = trans.shape
        eps = 1e-5
        ogfH, ogfW = (256, 704)
        reference_points = reference_points[None, None].repeat(B, N, 1, 1, 1, 1)
        reference_points = torch.inverse(bda).view(B, 1, 1, 1, 1, 3,
                                                   3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)  # bda逆变换
        reference_points -= trans.view(B, N, 1, 1, 1, 3)
        combine = rots.matmul(torch.inverse(intrins)).inverse()
        reference_points_cam = combine.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points_cam = torch.cat([reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps),
                                          reference_points_cam[..., 2:3]], 5
                                         )
        reference_points_cam = post_rots.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).squeeze(
            -1)
        reference_points_cam += post_trans.view(B, N, 1, 1, 1, 3)
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH
        # reference_points_cam: (B, N, 100, 100, 4, 3) 3:(u,v,d) 归一化后的uv坐标

        # 去掉图像以外的点，mask表示reference_points投影到相机平面上在图像范围内的点
        mask = (reference_points_cam[..., 2:3] > eps)
        mask = (mask & (reference_points_cam[..., 0:1] > eps)
                & (reference_points_cam[..., 0:1] < (1.0 - eps))
                & (reference_points_cam[..., 1:2] > eps)
                & (reference_points_cam[..., 1:2] < (1.0 - eps)))
        # mask (B,N,100,100,4,1) 每个batch，每个camera下100x100x4的01mask

        ################################# NEW !!! ################################
        grid  = reference_points_cam[..., :2].clone() # (B, N, H, W, D, 2)
        grid[..., 0] = grid[..., 0] * 2 - 1  # x方向归一化到 [-1,1]
        grid[..., 1] = grid[..., 1] * 2 - 1  # y方向归一化到 [-1,1]

        # 调整成 grid_sample 需要的维度: (B*N, H, W, D, 2) -> (B*N, HWD, 1, 2)
        grid = grid.permute(0, 1, 2, 3, 4, 5).reshape(B * N, -1, 1, 2)

        # 预处理 illu_map -> (B*N, 1, H, W)
        illu_map_flat = illu_map.reshape(B * N, 1, ogfH, ogfW)

        illu_weights = F.grid_sample(illu_map_flat, grid, align_corners=True) # (B*N, 1, HWD, 1)

        B, N, H, W, D, _ = reference_points_cam.shape
        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3, 4, 5).reshape(N, B, H * W, D, 3)
        illu_weights = illu_weights.view(N, B, H * W, D).unsqueeze(-1)
        mask = mask.permute(1, 0, 2, 3, 4, 5).reshape(N, B, H * W, D, 1).squeeze(-1)
        return reference_points, reference_points_cam[..., :2], mask, reference_points_cam[..., 2:3], illu_weights


    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling_with_illu_weight(self, reference_points, cam_params=None, illu_map=None):
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        B, N, _ = trans.shape
        eps = 1e-5
        ogfH, ogfW = (256, 704)
        reference_points = reference_points[None, None].repeat(B, N, 1, 1, 1, 1)
        reference_points = torch.inverse(bda).view(B, 1, 1, 1, 1, 3,
                                                   3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)  # bda逆变换
        reference_points -= trans.view(B, N, 1, 1, 1, 3)
        combine = rots.matmul(torch.inverse(intrins)).inverse()
        reference_points_cam = combine.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points_cam = torch.cat([reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps),
                                          reference_points_cam[..., 2:3]], 5
                                         )
        reference_points_cam = post_rots.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).squeeze(
            -1)
        reference_points_cam += post_trans.view(B, N, 1, 1, 1, 3)
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH
        # reference_points_cam: (B, N, 100, 100, 4, 3) 3:(u,v,d) 归一化后的uv坐标

        # 去掉图像以外的点，mask表示reference_points投影到相机平面上在图像范围内的点
        mask = (reference_points_cam[..., 2:3] > eps)
        mask = (mask & (reference_points_cam[..., 0:1] > eps)
                & (reference_points_cam[..., 0:1] < (1.0 - eps))
                & (reference_points_cam[..., 1:2] > eps)
                & (reference_points_cam[..., 1:2] < (1.0 - eps)))
        # mask (B,N,100,100,4,1) 每个batch，每个camera下100x100x4的01mask

        ################################# NEW !!! ################################
        reference_points_cam_scaled = reference_points_cam.clone()

        if self.down_sampled_illu_map:
            # illu_map降采样到 16x44
            illu_H, illu_W = 16, 44
            reference_points_cam_scaled[..., 0] *= illu_W  # 映射到 illu_mask 的宽度
            reference_points_cam_scaled[..., 1] *= illu_H  # 映射到 illu_mask 的高度
            # 投影到相机坐标系后的uv索引
            indices = reference_points_cam_scaled[..., :2].long()  # torch.Size([1, 6, 100, 100, 4, 2])
            indices[..., 0] = torch.clamp(indices[..., 0], 0, illu_W - 1)
            indices[..., 1] = torch.clamp(indices[..., 1], 0, illu_H - 1)
        else:
            reference_points_cam_scaled[..., 0] *= ogfW  # 映射到原始宽度 704
            reference_points_cam_scaled[..., 1] *= ogfH  # 映射到原始高度 256
            indices = reference_points_cam_scaled[..., :2].long()  # (B, N, H, W, D, 2)
            indices[..., 0] = torch.clamp(indices[..., 0], 0, ogfW - 1)  # 宽度限制在 [0, 703]
            indices[..., 1] = torch.clamp(indices[..., 1], 0, ogfH - 1)  # 高度限制在 [0, 255]


        # 将 indices 展平以便索引
        indices_flat = indices.view(B, N, -1, 2)  # shape: (B, N, H*W*D, 2)

        # 从illu_map中获得每个3D点对应的光照值
        illu_weights = illu_map[
            torch.arange(B)[:, None, None],  # B
            torch.arange(N)[None, :, None],  # N
            indices_flat[..., 1],  # (B, N, H*W*D), y坐标索引
            indices_flat[..., 0]  # (B, N, H*W*D), x坐标索引
        ]  # shape: (B, N, H*W*D)

        B, N, H, W, D, _ = reference_points_cam.shape
        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3, 4, 5).reshape(N, B, H * W, D, 3)
        illu_weights = illu_weights.view(N, B, H * W, D).unsqueeze(-1)
        mask = mask.permute(1, 0, 2, 3, 4, 5).reshape(N, B, H * W, D, 1).squeeze(-1)
        return reference_points, reference_points_cam[..., :2], mask, reference_points_cam[..., 2:3], illu_weights

    def get_illu_bev_weight(self, per_cam_mask_list, illu_weights):
        """
        Args:
            per_cam_mask_list: (N, B, bev_h*bev_w, num_points_pillar)
            illu_weights: (N, B,  bev_h*bev_w, num_points_pillar, 1)

        Returns:
            illu_weights_bev_avg_batch: (B, 1, bev_h, bev_w)
        """
        N, B, _, num_points_pillar = per_cam_mask_list.shape
        illu_weights_bev_avg_list = []
        for b in range(B):
            # 计算每个摄像头的掩码
            mask = per_cam_mask_list[:,b,...].reshape(6, 200, 200, num_points_pillar)  # (6, 200, 200, 8)
            mask_per_cam = mask.sum(-1) > 0  # (6, 200, 200)

            # 计算每个摄像头的权重
            illu_weights_bev = illu_weights[:,b,...].reshape(6, 200, 200, num_points_pillar)  # (6, 200, 200, 8)
            illu_weights_bev = illu_weights_bev.mean(-1)  # (6, 200, 200)
            illu_weights_bev = torch.where(
                mask_per_cam,
                illu_weights_bev,
                torch.tensor(0.0, dtype=illu_weights_bev.dtype, device=illu_weights_bev.device)
            )

            # 对所有摄像头的权重求和
            illu_weights_bev_all = illu_weights_bev.sum(axis=0)  # (200, 200)

            # 计算每个像素的摄像头覆盖次数
            count_all = mask_per_cam.sum(axis=0).float()  # (200, 200)
            count_all[count_all == 0] = 1.0  # 避免除以0

            # 计算加权平均值
            illu_weights_bev_avg = illu_weights_bev_all / count_all  # (200, 200)

            # illu_weights_bev_avg = torch.where(
            #     illu_weights_bev_avg == torch.tensor(0.0, dtype=illu_weights_bev.dtype, device=illu_weights_bev.device),
            #     torch.tensor(0.40, dtype=illu_weights_bev.dtype, device=illu_weights_bev.device),
            #     illu_weights_bev_avg
            # )

            # 保存结果到list
            illu_weights_bev_avg_list.append(illu_weights_bev_avg)

        illu_weights_bev_avg_batch = torch.stack(illu_weights_bev_avg_list,dim=0).unsqueeze(1) # (B,1,200,200)

        return illu_weights_bev_avg_batch

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                cam_params=None,
                gt_bboxes_3d=None,
                pred_img_depth=None,
                bev_mask=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5] - self.pc_range[2], dim='3d', bs=bev_query.size(1), device=bev_query.device,
            dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        illu_map = kwargs['illu_map'] # torch.Size([B*N, 1, 256, 704])
        if self.down_sampled_illu_map:
            illu_map = torch.nn.functional.interpolate(illu_map, size=(16, 44), mode='bilinear')

        _,_,H,W = illu_map.shape
        illu_map = illu_map.view(-1,6,H,W)

        if self.use_gird_sample:
            ref_3d, reference_points_cam, per_cam_mask_list, bev_query_depth, illu_weights = self.point_sampling_with_illu_weight_grid_sample(
                ref_3d, cam_params=cam_params, illu_map=illu_map)
        else:
            ref_3d, reference_points_cam, per_cam_mask_list, bev_query_depth,illu_weights = self.point_sampling_with_illu_weight(
                ref_3d, cam_params=cam_params,illu_map=illu_map )


        # 光照图在bev空间的权重
        illu_bev_weight = self.get_illu_bev_weight(per_cam_mask_list, illu_weights)
        illu_bev_weight = self.bev_learning(illu_bev_weight) # (B,1,H,W)

        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                prev_bev=prev_bev, # None
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                per_cam_mask_list=per_cam_mask_list,
                bev_mask=bev_mask,
                bev_query_depth=bev_query_depth,
                pred_img_depth=pred_img_depth,
                **kwargs)

            bev_query = output # (B,bev_h*bev_w,80)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output,illu_bev_weight


@TRANSFORMER_LAYER.register_module()
class BEVFormerEncoderLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels=512,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerEncoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) in {2, 4, 6}
        # assert set(operation_order) in set(['self_attn', 'norm', 'cross_attn', 'ffn'])

    @force_fp32()
    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                debug=False,
                bev_mask=None,
                bev_query_depth=None,
                per_cam_mask_list=None,
                lidar_bev=None,
                pred_img_depth=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'
        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    None,
                    None,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=bev_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spatial cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    bev_query_depth=bev_query_depth,
                    pred_img_depth=pred_img_depth,
                    bev_mask=bev_mask,
                    per_cam_mask_list=per_cam_mask_list,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query