import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou_loss


class GeneralizedBoxIOULoss(nn.Module):
    def __init__(self, reduction="sum"):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, y_pred, y_true):
        return generalized_box_iou_loss(y_pred, y_true, reduction=self.reduction)
    

class LightweightFaceLoss(nn.Module):
    def __init__(self, w_cls=1.0, w_box=1.0, w_kps=0.5):
        super().__init__()
        
        self.cls_criterion = nn.BCEWithLogitsLoss(reduction='sum')
        self.box_criterion = GeneralizedBoxIOULoss(reduction='sum')
        self.kps_criterion = nn.SmoothL1Loss(reduction='sum')
        
        self.w_cls = w_cls
        self.w_box = w_box
        self.w_kps = w_kps

    def forward(self, y_pred, y_true):
        loss_cls = 0.0
        loss_box = 0.0
        loss_kps = 0.0
        
        num_scales = len(y_pred['cls'])
        
        for i in range(num_scales):
            loss_cls += self.cls_criterion(y_pred['cls'][i], y_true['cls'][i])
            
            pos_mask = y_true['cls'][i] == 1
            if pos_mask.sum() > 0:
                loss_box += self.box_criterion(y_pred['box'][i][pos_mask], y_true['box'][i][pos_mask])
                loss_kps += self.kps_criterion(y_pred['kps'][i][pos_mask], y_true['kps'][i][pos_mask])

        total_loss = (self.w_cls * loss_cls) + (self.w_box * loss_box) + (self.w_kps * loss_kps)
        
        return {
            "total_loss": total_loss,
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_kps": loss_kps,
        }
