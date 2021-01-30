# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by HUANG JIAN (jian.huang@hdu.edu.cn)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np



class DAG_(nn.Module):
    def __init__(self):
        super(DAG_, self).__init__()
        self.features = None
        self.connect_model = None
        self.align_head = None
        self.zf = None
        self.criterion = nn.BCEWithLogitsLoss() 
        self.neck = None
        self.search_size = 255
        self.score_size = 25
        self.batch = 32 if self.training else 1
        self.grids()
 
    def feature_extractor(self, x, online=False):
        return self.features(x, online=online)

    def extract_for_online(self, x):
        xf = self.feature_extractor(x, online=True)
        return xf

    def connector(self, template_feature, search_feature):
        pred_score = self.connect_model(template_feature, search_feature)
        return pred_score

    def _quality_loss(self, quality, reg_weight, label):   
        quality = quality.view(-1)           
        reg_weight = reg_weight.view(-1)     
        label = label.view(-1)              
        pos = label.data.eq(1).nonzero().squeeze().cuda()  
        loss_pos = self._cls_loss(quality, reg_weight, pos)
        return loss_pos  


    def _cls_loss(self, pred, label, select):
        if len(select.size()) == 0: return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.criterion(pred, label)  # the same as tf version
    
    def _weighted_BCE(self, pred, label):
        pred = pred.view(-1)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze().cuda()
        neg = label.data.eq(0).nonzero().squeeze().cuda()
        loss_pos = self._cls_loss(pred, label, pos)
        loss_neg = self._cls_loss(pred, label, neg)
        return loss_pos * 0.5 + loss_neg * 0.5

    def _weighted_BCE_align(self, pred, label):
        pred = pred.view(-1)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze().cuda()
        neg = label.data.eq(0).nonzero().squeeze().cuda()

        loss_pos = self._cls_loss(pred, label, pos)
        loss_neg = self._cls_loss(pred, label, neg)

        return loss_pos * 0.5 + loss_neg * 0.5


    def _IOULoss(self, iou, weight=None):
   
        losses = -torch.log(iou)
  
        if weight is not None and weight.sum() > 0:
    
            return (losses * weight).sum() / weight.sum()
        else:
            if losses.numel() == 0 :
                return torch.zeros(1).mean()
            return losses.mean()




    def calc_iou(self, pred, target, weight):

        bbox_pred_flatten = pred.permute(0, 2, 3, 1).reshape(-1, 4)  
        reg_target_flatten = target.reshape(-1, 4) 
        reg_weight_flatten = weight.reshape(-1)  
        pos_inds = torch.nonzero(reg_weight_flatten > 0).squeeze(1) 
        iou_weight = torch.zeros_like(weight) 
        iou_weight_flatten = iou_weight.reshape(-1) 
        bbox_pred_flatten = bbox_pred_flatten[pos_inds]  
        reg_target_flatten = reg_target_flatten[pos_inds]  
 
        pred_left = bbox_pred_flatten[:, 0]
        pred_top = bbox_pred_flatten[:, 1]
        pred_right = bbox_pred_flatten[:, 2]
        pred_bottom = bbox_pred_flatten[:, 3]

        target_left = reg_target_flatten[:, 0]
        target_top = reg_target_flatten[:, 1]
        target_right = reg_target_flatten[:, 2]
        target_bottom = reg_target_flatten[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
       
        area_intersect = w_intersect * h_intersect

        area_union = target_area + pred_area - area_intersect
        iou = (area_intersect + 1.0) / (area_union + 1.0)   
        iou_weight_flatten[pos_inds] = iou
        
        return iou_weight_flatten, pos_inds, iou_weight

    def add_iouloss(self, iou_weight_flatten, reg_weight_, pos_inds):
       
        
        iou_weight = iou_weight_flatten[pos_inds]
        loss = self._IOULoss(iou_weight)       
        return loss


    # ---------------------
    # classification align
    # ---------------------
    def grids(self):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = self.score_size    # 25
        stride = 8

        sz_x = sz // 2          # 12
        sz_y = sz // 2          # 12

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))
        self.grid_to_search = {}
       
        self.grid_to_search_x = x * stride + self.search_size // 2 
        self.grid_to_search_y = y * stride + self.search_size // 2  

        self.grid_to_search_x = torch.Tensor(self.grid_to_search_x).unsqueeze(0).unsqueeze(0).cuda()    
        self.grid_to_search_y = torch.Tensor(self.grid_to_search_y).unsqueeze(0).unsqueeze(0).cuda()    

        self.grid_to_search_x = self.grid_to_search_x.repeat(self.batch, 1, 1, 1)                      
        self.grid_to_search_y = self.grid_to_search_y.repeat(self.batch, 1, 1, 1)                   

   
    def template(self, z):
        _, self.zf = self.feature_extractor(z)

        if self.neck is not None:
            _, self.zf = self.neck(self.zf, crop=True)
    

    def track(self, x):

        _, xf = self.feature_extractor(x)

        if self.neck is not None:
            xf = self.neck(xf)
        
        bbox_pred, cls_pred, _, _ ,quality_pred = self.connect_model(xf, self.zf)

        return cls_pred, bbox_pred, quality_pred


    def forward(self, template, search, label=None, reg_target=None):
        """
             labels label
             reg_target: l,b,t,r
        :return:
        """
        _, zf = self.feature_extractor(template)
        _, xf = self.feature_extractor(search)

        if self.neck is not None :
            _, zf = self.neck(zf, crop=True)
            xf = self.neck(xf, crop=False)

        bbox_pred, cls_pred, _, _ ,quality= self.connect_model(xf, zf)
        
        iou_weight_flatten_, pos_inds, iou_weight_ = self.calc_iou(bbox_pred, reg_target, label)
        reg_loss = self.add_iouloss(iou_weight_flatten_, label, pos_inds)  
        cls_loss = self._weighted_BCE(cls_pred, label)

        quality_loss = self._quality_loss(quality, iou_weight_, label)

        return cls_loss, None, reg_loss, quality_loss









