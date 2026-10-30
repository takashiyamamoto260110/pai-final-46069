"""Portable extraction of the project's ResVLA inference adapter.

Upstream ResVLA and model files must be installed separately. Not a new model.
"""
import random
import numpy as np


class Policy:
    def __init__(self, checkpoint, base_vlm, seed=42):
        import torch
        from transformers import Qwen3VLForConditionalGeneration
        from resVLA.model.framework.__init__ import build_framework
        from resVLA.model.framework.share_tools import read_mode_config,dict_to_namespace
        if not torch.cuda.is_available():raise RuntimeError('CUDA/ROCm GPU required')
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        original=Qwen3VLForConditionalGeneration.from_pretrained
        def portable(*a,**kw):
            kw.update(attn_implementation='sdpa',dtype=torch.bfloat16)
            return original(*a,**kw)
        Qwen3VLForConditionalGeneration.from_pretrained=portable
        try:
            cfg,stats=read_mode_config(checkpoint)
            cfg['framework']['qwenvl']['base_vlm']=str(base_vlm)
            cfg=dict_to_namespace(cfg);cfg.trainer.pretrained_checkpoint=None
            model=build_framework(cfg=cfg)
        finally:Qwen3VLForConditionalGeneration.from_pretrained=original
        state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        model.load_state_dict(state,strict=True,assign=True);model.norm_stats=stats
        model=model.to(device='cuda',dtype=torch.bfloat16).eval()
        if torch.version.hip:
            import torch.nn.functional as F
            proj=model.qwen_vl_interface.model.visual.patch_embed.proj
            weight=proj.weight.detach().float().cpu()
            bias=proj.bias.detach().float().cpu() if proj.bias is not None else None
            def cpu_conv(x):
                y=F.conv3d(x.detach().float().cpu(),weight,bias,proj.stride,proj.padding,proj.dilation,proj.groups)
                return y.to(device=x.device,dtype=x.dtype)
            proj.forward=cpu_conv
        self.model=model;self.chunk=8;self.cache=[]
        action=stats['franka']['action']
        self.low=np.asarray(action['min'],dtype=np.float32)
        self.high=np.asarray(action['max'],dtype=np.float32)
        self.mask=np.asarray(action['mask'],dtype=bool)

    def act(self,obs,instruction):
        import torch
        if not self.cache:
            images=[np.ascontiguousarray(obs[k][::-1,::-1],dtype=np.uint8)
                    for k in ['agentview_image','robot0_eye_in_hand_image']]
            with torch.inference_mode():
                output=self.model.predict_action(examples=[dict(image=images,lang=instruction)])
            a=np.clip(np.asarray(output['normalized_actions'],dtype=np.float32)[0,:self.chunk],-1,1)
            a[:,6]=(a[:,6]>=.5).astype(np.float32)
            scaled=.5*(a+1)*(self.high-self.low)+self.low
            a=np.where(self.mask,scaled,a);a[:,6]=1-2*(a[:,6]>.5)
            self.cache=list(a.astype(np.float32))
        action=self.cache.pop(0)
        if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Invalid action')
        return action
