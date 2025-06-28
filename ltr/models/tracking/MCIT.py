from re import search
import torch
import torch.nn as nn
import ltr.models.backbone as backbones
import ltr.models.fusion as fusions
import ltr.models.head as heads
from ltr import model_constructor
from pytracking import TensorDict
from util.box_ops import box_xyxy_to_cxcywh

class MCIT(nn.Module):
    """The MCIT network.
    args:
        backbone:  Backbone feature extractor network.
        head:  Bounding box prediction network."""

    def __init__(self, backbone, head, head_type):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.head_type = head_type
        
    def forward(self, data):
        """Runs the network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking."""
        out_s_rgb, out_s_aux = self.backbone(data['template_rgb_images'], data['template_aux_images'], data['search_rgb_images'], data['search_aux_images'])
        if self.head_type == 'ACENTER':
            # output_format='BNC'
            # fusion_feat = torch.concat((out_s_rgb, out_s_aux), dim=1)
            fusion_feat = out_s_rgb + out_s_aux
        else:
            # output_format='BCHW'
            fusion_feat = out_s_rgb + out_s_aux
        output = self._head_forward(fusion_feat)
        return output
    
    def track(self, template_rgb_images, template_aux_images, search_rgb_images, search_aux_images):
        """For tracking."""
        out_s_rgb, out_s_aux = self.backbone(template_rgb_images, template_aux_images, search_rgb_images, search_aux_images)
        if self.head_type == 'ACENTER':
            # output_format='BNC'
            # fusion_feat = torch.concat((out_s_rgb, out_s_aux), dim=1)
            fusion_feat = out_s_rgb + out_s_aux
        else:
            # output_format='BCHW'
            fusion_feat = out_s_rgb + out_s_aux
        output = self._head_forward(fusion_feat)
        return output
    
    def _head_forward(self, fusion_feat):
        output = {}
        if self.head_type in ['CENTER', 'ACENTER']:
            score_map, pred_boxes, size_map, offset_map  = self.head(fusion_feat)
            output['score_map'] = score_map
            output['pred_boxes'] = pred_boxes     # (cx, cy, w, h)
            output['size_map'] = size_map
            output['offset_map'] = offset_map
        elif self.head_type == 'CORNER':
            pred_boxes = self.head(fusion_feat)    # (x1, y1, x2, y2)
            pred_boxes = box_xyxy_to_cxcywh(pred_boxes)   # (cx, cy, w, h)
            output['pred_boxes'] = pred_boxes
        elif self.head_type == 'MLP':
            pred_boxes = self.head(fusion_feat)    # (cx, cy, w, h)
            output['pred_boxes'] = pred_boxes
        else:
            raise NotImplementedError
        return TensorDict(output)

@model_constructor
def vit_tracker(settings):
    # Backbone
    backbone = backbones.vit_backbone(model_name='vit_base_patch16',
                                    img_size_t=settings.template_sz, 
                                    img_size_s=settings.search_sz,
                                    output_format='BCHW')
    if settings.head_type == 'CENTER':
        head = heads.center_head(inplanes=settings.fusion_network_output_channels,
                                channel=settings.head_channels,
                                feat_sz=settings.search_feat_sz,
                                stride = settings.backbone_down_sampling)
    elif settings.head_type == 'CORNER':
        head = heads.coner_head(inplanes=settings.fusion_network_output_channels,
                                channel=settings.head_channels,
                                feat_sz=settings.search_feat_sz,
                                stride = settings.backbone_down_sampling)
    elif settings.head_type == 'MLP':
        head = heads.mlp_head(input_dim=768*2,
                              hidden_dim=768,
                              output_dim=384,
                              num_layers=2,
                              LN=True)
    elif settings.head_type == 'ACENTER':
        head = heads.attn_center_head(inplanes=settings.fusion_network_output_channels, 
                               channel=settings.head_channels, 
                               feat_sz=settings.search_feat_sz,
                               num_heads=8)
    else:
        raise NotImplementedError
    net = MCIT(backbone, head, settings.head_type)
    return net