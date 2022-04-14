from functools import partial

from einops.einops import rearrange,repeat
from torch import nn
import argparse
import os
import warnings

from mmcv import Config, DictAction
from mmcv.runner import  load_checkpoint
import sys
sys.path.insert(0,'models/swin_model')
from mmdet.models import build_detector
from icecream import ic
import torch
from models.trans_vg import MuModule


class QRNet(nn.Module):
    def __init__(self,args) -> None:
        super(QRNet,self).__init__()
        self.args=args
        self.flag=None
        config='models/swin_model/configs/my_config/simple_multimodal_fpn_config.py'
        self.flag='multi-modal'
        cfg_options=None

        checkpoint=args.swin_checkpoint
        cfg = Config.fromfile(config)
        if cfg_options is not None:
            cfg.merge_from_dict(cfg_options)
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])
        cfg.model.neck.type=args.soft_fpn
        cfg.model.backbone.use_spatial=args.use_spatial
        cfg.model.backbone.use_channel=args.use_channel
        cfg.model.neck.use_spatial=args.use_spatial
        cfg.model.neck.use_channel=args.use_channel
     
        
        self.cfg=cfg
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        self.pretrained_parameter_name=checkpoint.keys()  
        
        self.num_channels=cfg.model['neck']['out_channels']
        self.backbone=model.backbone
        self.neck=model.neck
        
        self.rpn_head=model.rpn_head

    
    def forward(self,img,mask,text=None,extra:dict=None):
        import torch.nn.functional as F
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img,text)

        x = self.neck(x,text)
        out_mask=[F.interpolate(mask[None].float(), size=_.shape[-2:]).to(torch.bool)[0] for _ in x]
    
        # flatten 
        shape=[_.shape[-2:] for _ in x] 
        x=[rearrange(_,'B C H W -> (H W) B C') for _ in x]
        out_mask=[rearrange(_,'B H W -> B (H W)') for _ in out_mask]
        return x,out_mask

if __name__=='__main__':
    pass
        
        

