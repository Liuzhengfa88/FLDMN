import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.da_faster_rcnn_instance_weight.DA import _ImageDA, _InstanceDA
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
from model.utils.config import cfg
from model.utils.net_utils import (
    _affine_grid_gen,
    _affine_theta,
    _crop_pool_layer,
    _smooth_l1_loss,
)
from torch.autograd import Variable

from model.da_faster_rcnn_instance_weight.nt_xent import NTXentLoss


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, in_channel=4096):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign(
            (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0
        )

        self.grid_size = (
            cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        )
        # self.RCNN_roi_crop = _RoICrop()

        self.RCNN_imageDA = _ImageDA(self.dout_base_model, self.n_classes)
        self.RCNN_instanceDA = _InstanceDA(in_channel)
        self.consistency_loss = torch.nn.MSELoss(size_average=False)
        self.conv_lst = nn.Conv2d(self.dout_base_model, self.n_classes - 1, 1, 1, 0)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        
        # projection MLP , zf
        self.s_l1 = nn.Linear(self.dout_base_model, self.dout_base_model)
        self.s_l2 = nn.Linear(self.dout_base_model, 128)
        self.s_nt_xent_criterion = NTXentLoss(batch_size=self.n_classes-1, temperature=0.5, use_cosine_similarity=True)
        

        self.contra_softmax = nn.Softmax(dim=1)

    def entropy_loss(self,v):
        """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
        """
        assert v.dim() == 4
        n, c, w, h= v.size()
        return -torch.sum(torch.mul(v, torch.log(v + 1e-30))) / (n *w*h* np.log2(c))

    def base_feat_map(self, x):
        x_flatten = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x_map = self.conv_lst(self.avg_pool(x)).squeeze(-1).squeeze(-1)
        x_map = self.contra_softmax(x_map)
        x_out = []
        for i in range(self.n_classes-1):
            ps = x_map[:,i].unsqueeze(1)
            local_x = ps * x_flatten
            x_out.append(local_x)
        return x_out

    def _step(self, xis, xjs):
    
        # get the representations and the projections
        ris = xis
        rjs = xjs
        
        # [N,C]
        zis = self.s_l1(ris)
        zis = F.relu(zis)
        zis = self.s_l2(zis)

        zjs = self.s_l1(rjs)
        zjs = F.relu(zjs)
        zjs = self.s_l2(zjs)
        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.s_nt_xent_criterion(zis, zjs)
        return loss


    def compute_contra_loss(self, x):
        x_map = self.base_feat_map(x)
        contra_loss = 0
        xis = x_map[0][0].unsqueeze(0)
        xjs = x_map[0][1].unsqueeze(0)
        for i in range(1, self.n_classes-1):
            xis = torch.cat([xis, x_map[i][0].unsqueeze(0)], dim=0)
            xjs = torch.cat([xjs, x_map[i][1].unsqueeze(0)], dim=0)

        contra_loss = self._step(xis, xjs)
        
        return contra_loss

    def KLDistance(self, s, t):
    
        # logp_x = F.log_softmax(x, dim=-1)
        # p_y = F.softmax(y, dim=-1)
        logs_x = F.log_softmax(s, dim=-1)
        s_x = F.softmax(s, dim=-1)

        logt_y = F.log_softmax(t, dim=-1)
        t_y = F.softmax(t, dim=-1)

        s_t = F.kl_div(logt_y, s_x, reduction='sum')
        t_s = F.kl_div(logs_x, t_y, reduction='sum')
        return (s_t + t_s)/2
        
    def forward(
        self,
        src_im_data,
        src_im_info,
        src_im_cls_lb,
        src_gt_boxes,
        src_num_boxes,
        src_need_backprop,
        _tgt_im_data,
        _tgt_im_info,
        _tgt_gt_boxes,
        _tgt_num_boxes,
        _tgt_need_backprop,
        ft_im_data,
        ft_im_info,
        ft_im_cls_lb,
        ft_gt_boxes,
        ft_num_boxes,
        ft_need_backprop,
        fs_im_data,
        fs_im_info,
        fs_gt_boxes,
        fs_num_boxes,
        fs_need_backprop,
        weight_value=1.0,
    ):


        #concate src_im_data and ft_im_data
        im_data = torch.cat([src_im_data, ft_im_data], dim=0)
        im_info = torch.cat([src_im_info, ft_im_info], dim=0)
        im_cls_lb = torch.cat([src_im_cls_lb, ft_im_cls_lb], dim=0)
        gt_boxes = torch.cat([src_gt_boxes, ft_gt_boxes], dim=0)
        num_boxes = torch.cat([src_num_boxes, ft_num_boxes], dim=0)
        need_backprop = torch.cat([src_need_backprop, ft_need_backprop], dim=0)

        batch_size = im_data.size(0)
        im_info = im_info.data
        im_cls_lb =  im_cls_lb.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        need_backprop = need_backprop.data

        base_feat = self.RCNN_base(im_data)
        cls_feat = self.conv_lst(self.avg_pool(base_feat)).squeeze(-1).squeeze(-1)
        img_cls_loss = nn.BCEWithLogitsLoss()(cls_feat, im_cls_lb)

        #for image-level contra loss
        contra_src_loss = self.compute_contra_loss(base_feat)


        #concate _tgt_data and fs_im_data
        tgt_im_data = torch.cat([_tgt_im_data, fs_im_data], dim=0)
        tgt_im_info = torch.cat([_tgt_im_info, fs_im_info], dim=0)
        tgt_gt_boxes = torch.cat([_tgt_gt_boxes, fs_gt_boxes], dim=0)
        tgt_num_boxes = torch.cat([_tgt_num_boxes, fs_num_boxes], dim=0)
        tgt_need_backprop = torch.cat([_tgt_need_backprop, fs_need_backprop], dim=0)

        tgt_batch_size = tgt_im_data.size(0)
        tgt_im_info = tgt_im_info.data
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data
        tgt_need_backprop = tgt_need_backprop.data

        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.train()
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(
            base_feat, im_info, gt_boxes, num_boxes
        )

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(
                rois_outside_ws.view(-1, rois_outside_ws.size(2))
            )

        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == "align":
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == "pool":
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(
                bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4
            )
            bbox_pred_select = torch.gather(
                bbox_pred_view,
                1,
                rois_label.view(rois_label.size(0), 1, 1).expand(
                    rois_label.size(0), 1, 4
                ),
            )
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        # ins_contra_loss = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(
                bbox_pred, rois_target, rois_inside_ws, rois_outside_ws
            )

        #for probability invariance
        invar_num = 60
        invar_index = np.random.choice(rois.size(1), size=invar_num)
        invar_rois = torch.zeros((rois.size(0) * invar_num, rois.size(2))).cuda()
        for i in range(batch_size):
            for j in range(invar_num):
                invar_rois[i*invar_num + j] = rois[i][invar_index[j]]
        invar_rois = torch.cat([invar_rois.unsqueeze(0), invar_rois.unsqueeze(0)], dim=0)

        if cfg.POOLING_MODE == "align":
            invar_pooled_feat = self.RCNN_roi_align(base_feat, invar_rois.view(-1, 5))
        elif cfg.POOLING_MODE == "pool":
            invar_pooled_feat = self.RCNN_roi_pool(base_feat, invar_rois.view(-1, 5))

        # feed pooled features to top model
        invar_pooled_feat = self._head_to_tail(invar_pooled_feat)
        # compute object classification probability
        invar_cls_score = self.RCNN_cls_score(invar_pooled_feat)
        invar_cls_prob = F.softmax(invar_cls_score, 1)
        invar_cls_prob = invar_cls_prob.view(batch_size, -1, invar_cls_prob.size(1))
        s_invar_cls_prob = invar_cls_prob[:1].squeeze(0)
        ft_invar_cls_prob = invar_cls_prob[1:].squeeze(0)
        invar_kdl_loss = self.KLDistance(s_invar_cls_prob, ft_invar_cls_prob)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        """ =================== for target =========================="""

        tgt_batch_size = tgt_im_data.size(0)
        tgt_im_info = (
            tgt_im_info.data
        )  # (size1,size2, image ratio(new image / source image) )
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data
        tgt_need_backprop = tgt_need_backprop.data

        # feed image data to base model to obtain base feature map
        tgt_base_feat = self.RCNN_base(tgt_im_data)

        contra_tgt_loss = self.compute_contra_loss(tgt_base_feat)


        tgt_img_cls_feat = self.conv_lst(tgt_base_feat)
        tgt_img_cls_feat = F.softmax(tgt_img_cls_feat, dim=1)
        tgt_img_cls_loss = self.entropy_loss(tgt_img_cls_feat)

        # add new code
        tgt_image_cls_feat = (
            self.conv_lst(self.avg_pool(tgt_base_feat)).squeeze(-1).squeeze(-1).detach()
        )
        # tgt_image_cls_feat = F.sigmoid(tgt_image_cls_feat[0]).detach()

        tgt_image_cls_feat = F.sigmoid(tgt_image_cls_feat).detach()
        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.eval()
        tgt_rois, tgt_rpn_loss_cls, tgt_rpn_loss_bbox = self.RCNN_rpn(
            tgt_base_feat, tgt_im_info, tgt_gt_boxes, tgt_num_boxes
        )

        # if it is training phrase, then use ground trubut bboxes for refining

        tgt_rois_label = None
        tgt_rois_target = None
        tgt_rois_inside_ws = None
        tgt_rois_outside_ws = None
        tgt_rpn_loss_cls = 0
        tgt_rpn_loss_bbox = 0

        tgt_rois = Variable(tgt_rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == "crop":
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            tgt_grid_xy = _affine_grid_gen(
                tgt_rois.view(-1, 5), tgt_base_feat.size()[2:], self.grid_size
            )
            tgt_grid_yx = torch.stack(
                [tgt_grid_xy.data[:, :, :, 1], tgt_grid_xy.data[:, :, :, 0]], 3
            ).contiguous()
            tgt_pooled_feat = self.RCNN_roi_crop(
                tgt_base_feat, Variable(tgt_grid_yx).detach()
            )
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                tgt_pooled_feat = F.max_pool2d(tgt_pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == "align":
            tgt_pooled_feat = self.RCNN_roi_align(tgt_base_feat, tgt_rois.view(-1, 5))
        elif cfg.POOLING_MODE == "pool":
            tgt_pooled_feat = self.RCNN_roi_pool(tgt_base_feat, tgt_rois.view(-1, 5))

        # feed pooled features to top model
        tgt_pooled_feat = self._head_to_tail(tgt_pooled_feat)

        # add new code
        tgt_cls_score = self.RCNN_cls_score(tgt_pooled_feat).detach()
        tgt_prob = F.softmax(tgt_cls_score, 1).detach()
        tgt_pre_label = tgt_prob.argmax(1).detach()




        #for probability invariance for target domain, zf
        tgt_invar_num = 60
        tgt_invar_index = np.random.choice(tgt_rois.size(1), size=tgt_invar_num)
        tgt_invar_rois = torch.zeros((tgt_rois.size(0) * tgt_invar_num, tgt_rois.size(2))).cuda()
        for i in range(tgt_batch_size):
            for j in range(tgt_invar_num):
                tgt_invar_rois[i*tgt_invar_num + j] = tgt_rois[i][tgt_invar_index[j]]
        tgt_invar_rois = torch.cat([tgt_invar_rois.unsqueeze(0), tgt_invar_rois.unsqueeze(0)], dim=0)
        
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == "align":
            tgt_invar_pooled_feat = self.RCNN_roi_align(tgt_base_feat, tgt_invar_rois.view(-1, 5))
        elif cfg.POOLING_MODE == "pool":
            tgt_invar_pooled_feat = self.RCNN_roi_pool(tgt_base_feat, tgt_invar_rois.view(-1, 5))

        # feed pooled features to top model
        tgt_invar_pooled_feat = self._head_to_tail(tgt_invar_pooled_feat)
        # compute object classification probability
        tgt_invar_cls_score = self.RCNN_cls_score(tgt_invar_pooled_feat)
        tgt_invar_cls_prob = F.softmax(tgt_invar_cls_score, 1)
        tgt_invar_cls_prob = tgt_invar_cls_prob.view(tgt_batch_size, -1, tgt_invar_cls_prob.size(1))
        t_invar_cls_prob = tgt_invar_cls_prob[:1].squeeze(0)
        fs_invar_cls_prob = tgt_invar_cls_prob[1:].squeeze(0)
        tgt_invar_kdl_loss = self.KLDistance(t_invar_cls_prob, fs_invar_cls_prob)


        """  DA loss   """

        # DA LOSS
        DA_img_loss_cls = 0
        DA_ins_loss_cls = 0

        tgt_DA_img_loss_cls = 0
        tgt_DA_ins_loss_cls = 0

        base_score, local_base_score, base_label = self.RCNN_imageDA(base_feat, need_backprop)

        # Image DA
        base_prob = F.log_softmax(base_score, dim=1)
        DA_img_loss_cls = F.nll_loss(base_prob, base_label)

        #Image DA for local
        local_DA_img_loss_cls = 0
        for i in range(self.n_classes-1):
            local_base_prob_i = F.log_softmax(local_base_score[i], dim=1)
            local_DA_img_loss_cls_i = F.nll_loss(local_base_prob_i, base_label)
            local_DA_img_loss_cls += local_DA_img_loss_cls_i
        
        local_DA_img_loss_cls = local_DA_img_loss_cls / (self.n_classes-1)

        instance_sigmoid, same_size_label = self.RCNN_instanceDA(
            pooled_feat, need_backprop
        )
        instance_loss = nn.BCELoss()
        DA_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)

        # # consistency_prob = torch.max(F.softmax(base_score, dim=1),dim=1)[0]
        # consistency_prob = F.softmax(base_score, dim=1)[:, 1, :, :]
        # consistency_prob = torch.mean(consistency_prob)
        # consistency_prob = consistency_prob.repeat(instance_sigmoid.size())

        # DA_cst_loss = self.consistency_loss(instance_sigmoid, consistency_prob.detach())

        #new consistency prob, zf
        DA_cst_loss = 0
        consistency_prob = F.softmax(base_score, dim=1)[:, 1, :, :]
        da_instance_sigmoid = instance_sigmoid.view(batch_size, -1,1)
        for i in range(batch_size):
            consistency_prob_i = torch.mean(consistency_prob[i])
            da_instance_sigmoid_i = da_instance_sigmoid[i]
            consistency_prob_i = consistency_prob_i.repeat(da_instance_sigmoid_i.size())
            DA_cst_loss_i = self.consistency_loss(da_instance_sigmoid_i, consistency_prob_i.detach())

            DA_cst_loss += DA_cst_loss_i
        DA_cst_loss = DA_cst_loss / batch_size


        """  ************** taget loss ****************  """

        tgt_base_score, tgt_local_base_score, tgt_base_label = self.RCNN_imageDA(
            tgt_base_feat, tgt_need_backprop
        )

        # Image DA
        tgt_base_prob = F.log_softmax(tgt_base_score, dim=1)
        tgt_DA_img_loss_cls = F.nll_loss(tgt_base_prob, tgt_base_label)

        tgt_instance_sigmoid, tgt_same_size_label = self.RCNN_instanceDA(
            tgt_pooled_feat, tgt_need_backprop
        )

        #Image DA for local
        tgt_local_DA_img_loss_cls = 0
        for i in range(self.n_classes-1):
            tgt_local_base_prob_i = F.log_softmax(tgt_local_base_score[i], dim=1)
            tgt_local_DA_img_loss_cls_i = F.nll_loss(tgt_local_base_prob_i, tgt_base_label)
            tgt_local_DA_img_loss_cls += tgt_local_DA_img_loss_cls_i
        
        tgt_local_DA_img_loss_cls = tgt_local_DA_img_loss_cls/(self.n_classes-1)

        # add new code
        target_weight = []
        tgt_rois_num_each = int(len(tgt_pre_label)/tgt_batch_size)
        tgt_image_cls_feat_index = -1
        for i in range(len(tgt_pre_label)):
            #zf
            if i % tgt_rois_num_each == 0:
                tgt_image_cls_feat_index +=1 

            label_i = tgt_pre_label[i].item()
            if label_i > 0:
                diff_value = torch.exp(
                    weight_value
                    * torch.abs(tgt_image_cls_feat[tgt_image_cls_feat_index][label_i - 1] - tgt_prob[i][label_i])
                ).item()
                target_weight.append(diff_value)
            else:
                target_weight.append(1.0)

        tgt_instance_loss = nn.BCELoss(
            weight=torch.Tensor(target_weight).view(-1, 1).cuda()
        )

        tgt_DA_ins_loss_cls = tgt_instance_loss(
            tgt_instance_sigmoid, tgt_same_size_label
        )

        # tgt_consistency_prob = F.softmax(tgt_base_score, dim=1)[:, 0, :, :]
        # tgt_consistency_prob = torch.mean(tgt_consistency_prob)
        # tgt_consistency_prob = tgt_consistency_prob.repeat(tgt_instance_sigmoid.size())

        # tgt_DA_cst_loss = self.consistency_loss(
        #     tgt_instance_sigmoid, tgt_consistency_prob.detach()
        # )

        #consistency_prob for batch, zf
        tgt_DA_cst_loss = 0
        tgt_consistency_prob = F.softmax(tgt_base_score, dim=1)[:, 0, :, :]
        tgt_da_instance_sigmoid = tgt_instance_sigmoid.view(tgt_batch_size, -1,1)
        for i in range(tgt_batch_size):
            tgt_consistency_prob_i = torch.mean(tgt_consistency_prob[i])
            tgt_da_instance_sigmoid_i = tgt_da_instance_sigmoid[i]
            tgt_consistency_prob_i = tgt_consistency_prob_i.repeat(tgt_da_instance_sigmoid_i.size())
            tgt_DA_cst_loss_i = self.consistency_loss(tgt_da_instance_sigmoid_i, tgt_consistency_prob_i.detach())

            tgt_DA_cst_loss += tgt_DA_cst_loss_i
        tgt_DA_cst_loss = tgt_DA_cst_loss / tgt_batch_size

        return (
            rois,
            cls_prob,
            bbox_pred,
            img_cls_loss,
            tgt_img_cls_loss,
            contra_src_loss,
            contra_tgt_loss,
            rpn_loss_cls,
            rpn_loss_bbox,
            RCNN_loss_cls,
            # tgt_RCNN_loss_cls,
            RCNN_loss_bbox,
            rois_label,
            invar_kdl_loss,
            tgt_invar_kdl_loss,
            DA_img_loss_cls,
            local_DA_img_loss_cls,
            DA_ins_loss_cls,
            tgt_DA_img_loss_cls,
            tgt_local_DA_img_loss_cls,
            tgt_DA_ins_loss_cls,
            DA_cst_loss,
            tgt_DA_cst_loss,
        )

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean
                )  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.conv_lst, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_imageDA.Conv1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_imageDA.Conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        for i in range(self.n_classes-1):
            normal_init(self.RCNN_imageDA.dci[i][0], 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.RCNN_imageDA.dci[i][2], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_instanceDA.dc_ip1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_instanceDA.dc_ip2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_instanceDA.clssifer, 0, 0.05, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
