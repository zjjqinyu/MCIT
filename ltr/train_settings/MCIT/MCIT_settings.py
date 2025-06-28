import os
import ltr.admin.settings as ws_settings
import torch

def get_tracker_settings(settings=None):
    if settings is None:
        settings = ws_settings.Settings()
    settings.description = 'Default train settings.'

    settings.multi_gpu = False
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    settings.device = torch.device("cuda:0")

    settings.batch_size = 24
    settings.num_workers = 8
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]

    settings.template_area_factor = 2.0
    settings.template_sz = 128
    settings.search_area_factor = 5.0
    settings.search_sz = 256

    settings.center_jitter_factor = {'template': 0, 'search': 4.0}
    settings.scale_jitter_factor = {'template': 0, 'search': 0.5}

    # Vit
    # settings.patch_size = 16
    # settings.num_layers = 12
    # settings.num_heads = 12
    # settings.hidden_dim = 768
    # settings.mlp_dim = 768
    # settings.dropout = 0.1
    # settings.attention_dropout = 0.1
    settings.backbone_down_sampling = 16

    settings.template_feat_sz = settings.template_sz//settings.backbone_down_sampling
    settings.search_feat_sz = settings.search_sz//settings.backbone_down_sampling

    settings.head_type = 'ACENTER'
    settings.head_channels = 256
    settings.fusion_network_output_channels = 768

    return settings