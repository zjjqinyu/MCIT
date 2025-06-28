import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models._manipulate import checkpoint_seq
from timm.models.vision_transformer import Attention
from timm.layers import get_norm_layer
from safetensors.torch import load_model
from safetensors.torch import load_file
from ltr.admin.environment import env_settings

class CrossAttention(Attention):
    def __init__(self, dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer):
        super().__init__(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer)

    def forward(self, x1, x2) -> torch.Tensor:
        B, N1, C = x1.shape
        qkv1 = self.qkv(x1).reshape(B, N1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q1, _, _ = qkv1.unbind(0)

        B, N2, C = x2.shape
        qkv2 = self.qkv(x2).reshape(B, N2, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        _, k2, v2 = qkv2.unbind(0)

        q1, k2 = self.q_norm(q1), self.k_norm(k2)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q1, k2, v2,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q1 = q1 * self.scale
            attn = q1 @ k2.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v2

        x = x.transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Cross-modal Interaction Module
class CMI_Module(nn.Module):
    def __init__(self, n_t, n_s, dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer, pretrained=True):
        super().__init__()
        self.n_t = n_t
        self.n_s = n_s
        self.self_attention = Attention(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer)
        self.cross_attention_rgb = CrossAttention(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer)
        self.cross_attention_aux = CrossAttention(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        tokens_rgb, tokens_aux = x
        z_rgb, x_rgb = torch.split(tokens_rgb, [self.n_t, self.n_s], dim=1)
        z_aux, x_aux = torch.split(tokens_aux, [self.n_t, self.n_s], dim=1)
        z = torch.concat((z_rgb, z_aux), dim=1)
        z = z + self.self_attention(self.ln(z))
        z_rgb, z_aux = torch.split(z, [self.n_t, self.n_t], dim=1)
        x_rgb = x_rgb + self.cross_attention_rgb(self.ln(x_rgb), self.ln(z))
        x_aux = x_aux + self.cross_attention_aux(self.ln(x_aux), self.ln(z))
        tokens_rgb = torch.concat((z_rgb, x_rgb), dim=1)
        tokens_aux = torch.concat((z_aux, x_aux), dim=1)
        return tokens_rgb, tokens_aux
    
_model_info = {
    'vit_base_patch16':{
        'model_args' : dict(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12),
        'download_page' : 'https://huggingface.co/timm/vit_base_patch16_384.augreg_in21k_ft_in1k/tree/main',
    }
}

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, img_size_t, img_size_s, output_format='BCHW', *args, **kwargs):
        assert output_format in ['BCHW', 'BNC']
        super().__init__(dynamic_img_size=True, *args, **kwargs)
        self.img_size_t = img_size_t
        self.img_size_s = img_size_s
        self.output_format = output_format
        self.n_t = (img_size_t//kwargs['patch_size'])**2
        self.n_s = (img_size_s//kwargs['patch_size'])**2

    def forward_embed(self, img_t, img_s):
        img_t = self.patch_embed(img_t)
        img_s = self.patch_embed(img_s)
        tokens_t = self._pos_embed(img_t)
        tokens_s = self._pos_embed(img_s)
        x = torch.concat((tokens_t, tokens_s), dim=1)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        return x
    
    def forward_post_proc(self, x):
        x = self.norm(x)
        B, N, C = x.size()
        out_t, out_s = torch.split(x, [self.n_t, self.n_s], dim=1)
        if self.output_format == 'BCHW':
            out_s = out_s.permute(0, 2, 1).reshape(B, C, int(self.n_s**0.5), int(self.n_s**0.5))   # BNC --> BCHW
        return out_s

    def forward(self, img_t, img_s):
        x = self.forward_embed(img_t, img_s)
        x = self.blocks(x)
        return self.forward_post_proc(x)

class VitBackbone(nn.Module):
    def __init__(self, model_name, img_size_t, img_size_s, im_layers_lst, pretrained=True, output_format='BCHW', *args, **kwargs): # , patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout
        super(VitBackbone, self).__init__()
        # im_layers_lst is the list of layer numbers for the interaction modules to be inserted.
        self.im_layers_lst = im_layers_lst
        self.vit = VisionTransformer(img_size_t, img_size_s, output_format, *args, **kwargs)
        if pretrained:
            pretrained_dir = env_settings().pretrained_networks + model_name
            pretrained_loc = pretrained_dir + '/model.safetensors'
            assert os.path.exists(pretrained_loc), f'The pretrained networks {model_name} was not found. You can download "model.safetensors" from {_model_info[model_name]["download_page"]} and place it in {pretrained_dir}'
            load_model(self.vit, pretrained_loc, strict=False)
        # remove cls_token
        self.vit.no_embed_class = True
        self.vit.cls_token = None
        self.vit.pos_embed = nn.Parameter(self.vit.pos_embed[:,1:,:])
        # Build interaction modules
        self.interaction_modules = nn.ModuleList([nn.Identity() for _ in range(kwargs['depth'])])
        n_t = (img_size_t//kwargs['patch_size'])**2
        n_s = (img_size_s//kwargs['patch_size'])**2
        for i in im_layers_lst:
            self.interaction_modules[i] = CMI_Module(n_t=n_t,
                                                            n_s=n_s,
                                                            dim=kwargs['embed_dim'],
                                                            num_heads=kwargs['num_heads'], 
                                                            qkv_bias=kwargs.get('qkv_bias', True), 
                                                            qk_norm=kwargs.get('qk_norm', False), 
                                                            attn_drop=kwargs.get('attn_drop_rate', 0), 
                                                            proj_drop=kwargs.get('proj_drop_rate', 0), 
                                                            norm_layer=get_norm_layer(kwargs.get('norm_layer', None)))
            if pretrained:
                new_dict = {}
                assert os.path.exists(pretrained_loc), f'The pretrained networks {model_name} was not found. You can download "model.safetensors" from {_model_info[model_name]["download_page"]} and place it in {pretrained_dir}'
                state_dict = load_file(pretrained_loc)
                for key, value in state_dict.items():
                    if f'blocks.{i}.attn.' in key:
                        new_key = key.replace(f'blocks.{i}.attn.', '')
                        new_dict[new_key] = value
                self.interaction_modules[i].self_attention.load_state_dict(new_dict)
                self.interaction_modules[i].cross_attention_aux.load_state_dict(new_dict)
                self.interaction_modules[i].cross_attention_rgb.load_state_dict(new_dict)
        
    def forward(self, img_t_rgb, img_t_aux, img_s_rgb, img_s_aux):
        tokens_rgb = self.vit.forward_embed(img_t_rgb, img_s_rgb)
        tokens_aux = self.vit.forward_embed(img_t_aux, img_s_aux)
        for i, block in enumerate(self.vit.blocks):
            tokens_rgb = block(tokens_rgb)
            tokens_aux = block(tokens_aux)
            tokens_rgb, tokens_aux = self.interaction_modules[i]((tokens_rgb, tokens_aux))
        out_s_rgb = self.vit.forward_post_proc(tokens_rgb)
        out_s_aux = self.vit.forward_post_proc(tokens_aux)
        return out_s_rgb, out_s_aux

def vit_backbone(model_name, img_size_t, img_size_s, output_format, pretrained=True):
    im_layers_lst = [1,5,9,12] # Define the index of the first layer as 1, which is the same as the paper.
    im_layers_lst2 = [num-1 for num in im_layers_lst] # In ViT, the actual index of the first layer is 0, so num-1 is required here.
    assert model_name in _model_info, 'The variable model_name is incorrect!'
    model_args = _model_info[model_name]['model_args']
    model = VitBackbone(model_name, img_size_t, img_size_s, im_layers_lst2, pretrained, output_format, **dict(model_args))
    return model