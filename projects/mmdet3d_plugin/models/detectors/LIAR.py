# Copyright (c) Phigent Robotics. All rights reserved.
from ...ops import TRTBEVPoolv2
from .bevdet import BEVDet
from .bevstereo4d import BEVStereo4D
from mmdet3d.models import DETECTORS, build_neck, build_backbone
from mmdet3d.models.builder import build_head,build_loss
import torch.nn.functional as F
import torch.nn as nn
import torch
CNT=0

@DETECTORS.register_module()
class LIAR(BEVDet):
    def __init__(self,
                # 光照增强模块
                 light_enhance=dict(
                                    type='Light_Enhance_V3',
                                    enhance_layer=1,
                                    calibrate_layer=3,
                                    enhance_channel=1,
                                    calibrate_channel=16,
                                    loss_light=dict(
                                        type='SCI_Loss',
                                        l2_loss_weight=0.5)),
                 light_enhance_weights_path=r"ckpts/light_enhance_new_e32.pth",
                 select_enhance=None,

                 fuse=None, # 光照融合模块
                 backward_projection=None, # BEVFormer components
                 
                 occ_head=None,
                 upsample=False,
                 **kwargs):
        super(LIAR, self).__init__(**kwargs)
        ##################################################################
        self.light_enhance = build_neck(light_enhance)
        if light_enhance_weights_path:
            light_enhance_weights = torch.load(light_enhance_weights_path)
            self.light_enhance.load_state_dict(light_enhance_weights)
            for param in self.light_enhance.parameters():
                param.requires_grad = False
        else:
            raise ValueError("light_enhance_weights_path !!!")

        self.gamma_corr = select_enhance
        if self.gamma_corr is not None:
            self.correction_module = build_neck(select_enhance)

        self.use_fuse = fuse
        if self.use_fuse is not None:
            self.fuse = build_neck(fuse)

        # BEVFormer init
        self.backward_projection = build_head(backward_projection)
        ##################################################################

        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample

    def denormalize(self,tensor,mean=[123.675, 116.28, 103.53],std=[58.395, 57.12, 57.375]):
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device).to(tensor.dtype)
        std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device).to(tensor.dtype)
        denorm_tensor = tensor * std + mean
        return denorm_tensor

    def normalize(self,tensor,mean=[123.675, 116.28, 103.53],std=[58.395, 57.12, 57.375]):
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device).to(tensor.dtype) # (1,3,1,1)
        std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device).to(tensor.dtype) # (1,3,1,1)
        norm_tensor = (tensor - mean) / std
        return norm_tensor

    # new: 在原来的encoder之前增加了光照增强模块
    def image_encoder(self, img, gray,stereo=False):
        """
        Args:
            img: (B, N, 3, H, W)
            gray: (B, N, 1, H, W) / [(B, N, 1, H, W)] test的时候会变成list，需要加判断
            stereo: bool
        Returns:
            x: (B, N, C, fH, fW)
            stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo) / None
        """

        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)

        # 先将imgs反归一化
        img_denorm = self.denormalize(imgs)

        if type(gray)==list:
            grays = gray[0]
        else:
            grays = gray
        grays = grays.view(B * N, 1, imH, imW)

        # 把灰度图输入到light enhance中，得到光照图 ill_first
        ilist, rlist, inlist, attlist = self.light_enhance(grays) 
        ill_first = ilist[0]

        # retinex，得到原图的增强图像 img_deillu: 0-255
        img_deillu = img_denorm / ill_first
        img_deillu = torch.clamp(img_deillu, 0,255)

        # 自适应增强
        if self.gamma_corr is not None:
            img_deillu = self.correction_module(img_denorm,img_deillu,ill_first) # 还是在0-255范围内


        # ###################  debug  ###################
        # import numpy as np
        # import os
        # global CNT
        # folder = r'/opt/data/private/test/ideals/nightocc/basemodel_clean/rgb_illu_npz'
        # file_name = os.path.join(folder,str(CNT)+'.npz')
        # np.savez(
        #     file_name,
        #     ill_first = ill_first.cpu().numpy(),
        #     img_denorm = img_denorm.cpu().numpy(),
        #     img_deillu = img_deillu.cpu().numpy()
        # )
        # CNT = CNT + 1
        # ###################  debug  ###################


        # 重新norm
        img_deillu = self.normalize(img_deillu) #(B*N,3,256,704)

        # 输入到backbone
        x = self.img_backbone(img_deillu) #[(B,C*16,fH,fW) ,(B,C*32,fH/2,fW/2)] tuple

        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)

        # ilist：每个stage的光照图
        return x, stereo_feat, ill_first

    # new: 原来函数修改过了。 return多加了一个enhance_output，在 view_transformation 模块中额外传入了 enhance_output（为了获得光照图）
    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        """ Extract features of images.
        img_inputs:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N, D, fH, fW)
            enhance_output: 光照增强模块的output
        """
        img_inputs = self.prepare_inputs(img_inputs)

        # 光照增强后的结果输入到encoder,并且返回光照图
        x, _ , illu_map= self.image_encoder(img_inputs[0],kwargs['img_gray'])  # x: (B, N, C, fH, fW)

        # encoder输出的提取后特征和光照图融合
        if self.use_fuse is not None:
            x = self.fuse(x, illu_map) # (B,N,C,16,44)

        bev_feat, context_feat, depth = self.img_view_transformer([x] + img_inputs[1:7])
        # import pdb;pdb.set_trace()
        # reshape to (B,N,C,H,W)
        _,C,H,W = context_feat.shape # 和transformer中模块对齐
        # import pdb;pdb.set_trace()
        

        _,Cd,_,_ = depth.shape

        if self.backward_projection is not None:
            bev_feat_refined,illu_bev_weight = self.backward_projection(
                mlvl_feats = [context_feat.view(-1, 6, C, H, W)],
                pred_img_depth = depth.view(-1, 6, Cd, H, W),
                img_inputs = img_inputs,
                lss_bev = bev_feat,
                illu_map = illu_map
            )
        
        illu_bev_weight = torch.clamp(illu_bev_weight, min=0.0, max=1.0) # (B,1,200,200)

        bev_feat = bev_feat_refined*(illu_bev_weight+0.50)+bev_feat

        x = self.bev_encoder(bev_feat)
        return [x], depth

    # new: 根据extract_img_feature修改，return多了一个 enhance_output
    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        """Extract features from images and points."""
        """
        points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
        img_inputs:
                imgs:  (B, N_views, 3, H, W)        
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        """
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = None

        return img_feats, pts_feats, depth

    # new:
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()

        voxel_semantics = kwargs['voxel_semantics']  # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']  # (B, Dx, Dy, Dz)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)
        return losses


    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    # new:
    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, _, = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        if not hasattr(self.occ_head, "get_occ_gpu"):
            occ_preds = self.occ_head.get_occ(outs, img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        else:
            occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        # import pdb;pdb.set_trace()
        device = img_inputs[0].device
        kwargs['img_gray'] = torch.rand(1, 6, 1, 256, 704,device=device)
        # import pdb;pdb.set_trace()
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs





@DETECTORS.register_module()
class LIAR_stereo(BEVStereo4D):
    def __init__(self,
                 #####################################################
                 # 光照增强模块
                 light_enhance=dict(
                     type='Light_Enhance_V3',
                     enhance_layer=1,
                     calibrate_layer=3,
                     enhance_channel=1,
                     calibrate_channel=16,
                     loss_light=dict(
                         type='SCI_Loss',
                         l2_loss_weight=0.5)),
                 light_enhance_weights_path=r"ckpts/light_enhance_new_e32.pth",
                 gamma_correction=None,

                 fuse=None,  # 光照融合模块
                 #####################################################
                 occ_head=None,
                 upsample=False,
                 **kwargs):
        super(LIAR_stereo, self).__init__(**kwargs)
        ##############################################################################
        self.light_enhance = build_neck(light_enhance)
        if light_enhance_weights_path:
            light_enhance_weights = torch.load(light_enhance_weights_path)
            self.light_enhance.load_state_dict(light_enhance_weights)
            for param in self.light_enhance.parameters():
                param.requires_grad = False
        else:
            raise ValueError("light_enhance_weights_path !!!")

        self.gamma_corr = gamma_correction
        if self.gamma_corr is not None:
            self.correction_module = build_neck(gamma_correction)

        self.use_fuse = fuse
        if self.use_fuse is not None:
            self.fuse = build_neck(fuse)
        ##################################################################

        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample

    def denormalize(self, tensor, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device).to(tensor.dtype)
        std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device).to(tensor.dtype)
        denorm_tensor = tensor * std + mean
        return denorm_tensor

    def normalize(self, tensor, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device).to(tensor.dtype)  # (1,3,1,1)
        std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device).to(tensor.dtype)  # (1,3,1,1)
        norm_tensor = (tensor - mean) / std
        return norm_tensor



    # new: 经过光照增强模块得到光照图，将原图增强后输入到2D backbone，最后经过光照融合模块
    # x, stereo_feat, ill_first
    def image_encoder(self, img, gray, stereo=False):
        """
        Args:
            img: (B, N, 3, H, W)
            gray: (B, N, 1, H, W) / [(B, N, 1, H, W)] test的时候会变成list，需要加判断
            stereo: bool
        Returns:
            x: (B, N, C, fH, fW)
            stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo) / None
        """

        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)

        # 光照增强模块
        img_denorm = self.denormalize(imgs)

        if type(gray) == list:
            grays = gray[0]
        else:
            grays = gray
        grays = grays.view(B * N, 1, imH, imW)
        ilist, rlist, inlist, attlist = self.light_enhance(grays)
        illu_map = ilist[0]
        img_deillu = img_denorm / illu_map
        img_deillu = torch.clamp(img_deillu, 0, 255)
        if self.gamma_corr is not None:
            img_deillu = self.correction_module(img_denorm, img_deillu, illu_map)  # 还是在0-255范围内
        img_deillu = self.normalize(img_deillu)  # (B*N,3,256,704)


        x = self.img_backbone(img_deillu)  # [(B,C*16,fH,fW) ,(B,C*32,fH/2,fW/2)] tuple

        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)

        # 光照融合模块
        x = self.fuse(x, illu_map)  # (B,N,C,16,44)

        return x, stereo_feat, illu_map

    # new: img_view_transformer 多传入光照图
    def prepare_bev_feat(self, img, sensor2keyego, ego2global, intrin,
                         post_rot, post_tran, bda, mlp_input, feat_prev_iv,
                         k2s_sensor, extra_ref_frame, gray):
        """
        Args:
            img:  (B, N_views, 3, H, W)
            sensor2keyego: (B, N_views, 4, 4)
            ego2global: (B, N_views, 4, 4)
            intrin: (B, N_views, 3, 3)
            post_rot: (B, N_views, 3, 3)
            post_tran: (B, N_views, 3)
            bda: (B, 3, 3)
            mlp_input: (B, N_views, 27)
            feat_prev_iv: (B*N_views, C_stereo, fH_stereo, fW_stereo) or None
            k2s_sensor: (B, N_views, 4, 4) or None
            extra_ref_frame:
        Returns:
            bev_feat: (B, C, Dy, Dx)
            context_feat: (B, C, fH, fW)
            depth: (B*N, D, fH, fW)
            stereo_feat: (B*N_views, C_stereo, fH_stereo, fW_stereo)
        """
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)     # (B*N_views, C_stereo, fH_stereo, fW_stereo)
            return None, None, stereo_feat
        # x: (B, N_views, C, fH, fW)
        # stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo)
        x, stereo_feat,illu_map = self.image_encoder(img, gray,stereo=True)

        # 建立cost volume 所需的信息.
        metas = dict(k2s_sensor=k2s_sensor,     # (B, N_views, 4, 4)
                     intrins=intrin,            # (B, N_views, 3, 3)
                     post_rots=post_rot,        # (B, N_views, 3, 3)
                     post_trans=post_tran,      # (B, N_views, 3)
                     frustum=self.img_view_transformer.cv_frustum.to(x),    # (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                     cv_downsample=4,
                     downsample=self.img_view_transformer.downsample,
                     grid_config=self.img_view_transformer.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat]
                     )
        # bev_feat: (B, C * Dz(=1), Dy, Dx)
        # context_feat: (B*N, C, fH, fW)
        # depth: (B * N, D, fH, fW)
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], metas,illu_map)
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]    # (B, C, Dy, Dx)

        return bev_feat, depth, stereo_feat

    # 对pipeline中的gray进行处理，重新构造shape
    def prepare_gray_images(self, gray):
        """
        Args:
            imgs:  (B, N, 3, H, W)        # N = 6 * (N_history + 1)

        Returns:
            imgs: List[(B, N_views, C, H, W), (B, N_views, C, H, W), ...]       len = N_frames
        """
        # test 时，gray是list
        if type(gray)==list:
            gray = gray[0]

        B, N, C, H, W = gray.shape
        N = N // self.num_frame     # N_views = 6
        gray = gray.view(B, N, self.num_frame, C, H, W)    # (B, N_views, N_frames, C, H, W)
        gray = torch.split(gray, 1, 2)
        gray = [t.squeeze(2) for t in gray]     # List[(B, N_views, C, H, W), (B, N_views, C, H, W), ...]
        return gray

    # new: img_view_transformer 返回 bev_feat, context_feat, depth输入到backward_projection
    def extract_img_feat(self,
                         img_inputs,
                         img_metas,
                         pred_prev=False,
                         sequential=False, # https://github.com/Yzichen/FlashOCC/issues/73
                         **kwargs):
        """
        Args:
            img_inputs:
                imgs:  (B, N, 3, H, W)        # N = 6 * (N_history + 1)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins:     (B, N, 3, 3)
                post_rots:   (B, N, 3, 3)
                post_trans:  (B, N, 3)
                bda_rot:  (B, 3, 3)
            img_metas:
            **kwargs:
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N_views, D, fH, fW)
        """
        if sequential:
            return self.extract_img_feat_sequential(img_inputs, kwargs['feat_prev'])

        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img_inputs, stereo=True)

        gray = kwargs['img_gray']
        gray = self.prepare_gray_images(gray)

        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        context_key_frame = None
        illumap_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame-1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame-self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)   # (B, N_views, 27)

                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)

                if key_frame:
                    bev_feat, depth, feat_curr_iv = self.prepare_bev_feat(*inputs_curr,gray)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = self.prepare_bev_feat(*inputs_curr,gray)


                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)

                if not key_frame:
                    feat_prev_iv = feat_curr_iv

        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1      # batch_size = 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)

            # (1, N_views, 4, 4) --> (N_prev, N_views, 4, 4)
            ego2globals_curr = \
                ego2globals[0].repeat(self.num_frame - 2, 1, 1, 1)
            # (1, N_views, 4, 4) --> (N_prev, N_views, 4, 4)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat(self.num_frame - 2, 1, 1, 1)
            ego2globals_prev = torch.cat(ego2globals[1:-1], dim=0)            # (N_prev, N_views, 4, 4)
            sensor2keyegos_prev = torch.cat(sensor2keyegos[1:-1], dim=0)      # (N_prev, N_views, 4, 4)
            bda_curr = bda.repeat(self.num_frame - 2, 1, 1)     # (N_prev, 3, 3)

            return feat_prev, [imgs[0],     # (1, N_views, 3, H, W)
                               sensor2keyegos_curr,     # (N_prev, N_views, 4, 4)
                               ego2globals_curr,        # (N_prev, N_views, 4, 4)
                               intrins[0],          # (1, N_views, 3, 3)
                               sensor2keyegos_prev,     # (N_prev, N_views, 4, 4)
                               ego2globals_prev,        # (N_prev, N_views, 4, 4)
                               post_rots[0],    # (1, N_views, 3, 3)
                               post_trans[0],   # (1, N_views, 3, )
                               bda_curr,       # (N_prev, 3, 3)
                               feat_prev_iv,
                               curr2adjsensor[0]]

        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) == 4:
                b, c, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]

        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame-2):
                bev_feat_list[adj_id] = self.shift_feature(
                    bev_feat_list[adj_id],  # (B, C, Dy, Dx)
                    [sensor2keyegos[0],     # (B, N_views, 4, 4)
                     sensor2keyegos[self.num_frame-2-adj_id]],  # (B, N_views, 4, 4)
                    bda  # (B, 3, 3)
                )   # (B, C, Dy, Dx)

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_key_frame



    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        """Extract features from images and points."""
        """
        points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
        img_inputs:
                imgs:  (B, N_views, 3, H, W)        
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        """

        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = None
        return img_feats, pts_feats, depth


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']  # (B, N_views, img_H, img_W)
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        voxel_semantics = kwargs['voxel_semantics']  # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']  # (B, Dx, Dy, Dz)
        loss_occ = self.forward_occ_train(img_feats[0], voxel_semantics, mask_camera)
        losses.update(loss_occ)
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        # kwargs: dict_keys(['gt_depth', 'voxel_semantics', 'mask_lidar', 'mask_camera', 'canvas'])
        # kwargs['canvas'][0][id][0]: (256,704,3)
        # kwargs['gt_depth'][0][0][id]: (256,704)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)

        occ_list = self.simple_test_occ(img_feats[0], img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        # occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        occ_preds = self.occ_head.get_occ(outs, img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds


    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        # import pdb;pdb.set_trace()
        device = img_inputs[0].device
        # kwargs['img_gray'] = torch.rand(1, 6, 1, 256, 704,device=device)

        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs