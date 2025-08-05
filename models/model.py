from __future__ import annotations

"""Optimised HashFormer A multimodal moderator Written by <Anynomouse>.

* graceful **None** handling for any missing modality.
* image‑to‑tokens defined once.
* losses are computed **only** for modalities that are present; missing branches contribute 0.
* removed redundant conversions and Debug prints.
* MultiheadAttention flattened bug fixed.
"""

import math
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.masked_encoder import MaskEncoder as MaskEncoder, TokenTypes, Attention
import lpips
from ptflops import get_model_complexity_info


class CL(torch.nn.Module):
    def __init__(self, config=None, bit=64):
        super(CL, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.bit = 64


    def forward(self, h1, h2, weighted):
        try:
            # print(h1.shape, weighted.shape)
            # exit()
            logits = torch.einsum('ik,jk->ij', h1, h2)
            logits = logits / self.bit / 0.3

            balance_logits = h1.sum(0) / h1.size(0)
            # reg = self.mse(balance_logits, torch.zeros_like(balance_logits)) - self.mse(h1, torch.zeros_like(h1))
            
            # reg_weight = 1e-3
            loss = self.ce(logits, weighted)# + reg_weight * reg
        except Exception as e:
            print(f"Error: {e}")
            loss = torch.tensor(1.0, requires_grad=True).to(h1.device)
        return loss

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=0.1, alpha=0.5, use_logits=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.use_logits = use_logits
        self.label_smoothing = 0.1

    def forward(self, logits, targets):
        # BCE per sample
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.use_logits:
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        else:
            bce = F.binary_cross_entropy(logits, targets, reduction='none')

        # p_t = exp(−BCE)  gives probability of the true class
        pt = torch.exp(-bce)

        # α_t: alpha for positives, (1−alpha) for negatives
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # focal term
        focal_term = (1 - pt) ** self.gamma

        loss = alpha_t * focal_term * bce
        return loss.mean()




class HashFormer(nn.Module):
    def __init__(self, *, dim: int = 512, device: torch.device | str = "cpu",pretraining=True):
        super().__init__()
        self.device = torch.device(device)
        self.pretraining = pretraining
        self.encoder = MaskEncoder(dim=dim, depth=6, heads=4, num_fusion_tokens=16, video_temporal_patch_size=4)
        bit=64
        self.landmark_projection = nn.Sequential(
            nn.Linear(478*2,512)
            )
        self.fc_norm = nn.LayerNorm(dim)
        self.fusion_attn = nn.MultiheadAttention(dim, 4, batch_first=True, device=self.device)
       
        self.proj_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.2),
            nn.LayerNorm(dim ),
            nn.LeakyReLU(), 
            nn.Linear(dim  , dim)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(dim , dim),  
            # nn.BatchNorm1d(dim ), 
            nn.LayerNorm(dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),    
            nn.Linear(dim , dim ),
            nn.Dropout(0.2),    
            nn.Linear(dim , 1)     
        )
        self.predictor_e = nn.Sequential(
            nn.Linear(dim, dim)
        )
        
        self.norm_pix_loss = True
        self.contrastive = CL()
        self.cls_loss = BinaryFocalLoss()
        self.cls = nn.BCEWithLogitsLoss()
      

    # --------------------------------------------------------------------- util
    def _maybe_tokens(self, images: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if not present(images) else self.image_to_tokens(images)
  # ---- safe pooling (no tokens → zeros) ---- #
    def _safe_mean(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            # zero vector in normalised space
            return torch.zeros(x.size(0), x.size(2), device=x.device)

        return x.mean(dim=1)

        
    def forward_mse_loss(self, target, pred):
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / torch.sqrt(var + 1e-6)
        
        loss = (pred - target).pow(2)
        loss = loss.mean()
        return loss

        
    def half_masking(self,x: torch.Tensor):
        B, D = x.shape
        split = D // 2
        x_ctx = x.clone()
        x_ctx[:, split:] = 0        # zero out second half
        mask = torch.zeros_like(x)  # [B, D]
        mask[:, split:] = 1.0       # 1.0 on the masked dims
        return x_ctx, mask

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        mfcc: Optional[torch.Tensor] = None,
        mfcc_aug: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        audio_aug: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,  # video is mandatory here
        video_aug: Optional[torch.Tensor] = None,
        landmarks: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        images_aug: Optional[torch.Tensor] = None,
        multi_label: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and total loss."""
        img_tok = None
        if video is not None: 
            video = video.permute(0,2,1,3,4)
        if video_aug is not None: 
            video_aug = video_aug.permute(0,2,1,3,4)
        # encode
        if landmarks is not None:
            landmarks = self.landmark_projection(torch.flatten(landmarks, start_dim=2, end_dim=3))

        pooled, af_tok, a_tok,vf_tok, v_tok, lf_tok, l_tok, logits= self.encoder(
            audio=audio,
            video=video, 
            landmarks=landmarks
        )
        
        
        audio_weighted, h1_audio, h2_audio, audio_video_weighted, h1_audio_video, h2_audio_video,video_weighted, h1_video, h2_video, video_audio_weighted, h1_video_audio, h2_video_audio, landmark_weighted, h1_landmark, h2_landmark, landmark_video_weighted, h1_landmark_video, h2_landmark_video = logits
       
        # if self.training:
        pooled_b, af_tok_b, a_tok_b,vf_tok_b, v_tok_b, lf_tok_b, l_tok_b, logits_b= self.encoder(
                audio=audio_aug,
                video=video_aug,
                landmarks=landmarks
            )
            
        if not self.pretraining:
            f_out, _ = self.fusion_attn(pooled, pooled, pooled)
            f_out = self.fc_norm(f_out) 
            assert torch.isfinite(f_out).all(), "fva_out has NaNs"
            avf_out = self._safe_mean(f_out)
            assert torch.isfinite(avf_out).all(), "fva_out has NaNs"
            logits_cls = self.proj_head(avf_out)
            logits_cls = self.cls_head(logits_cls)
        else:
            logits_cls = None
        

        tokens: Dict[str, torch.Tensor] = {
            "a": h1_audio,
            "v": h1_video,
            "l": h1_landmark,
            }
        mean_toks_intra = {
            "a":{
                "h1":h1_audio, 
                "h2":h2_audio,
                "embed":a_tok,
                "weighted":audio_weighted,
                "modalities_present":mfcc is not None 
            },
            "v":{
                "h1":h1_video, 
                "h2":h2_video,
                "embed":v_tok,
                "weighted":video_weighted,
                "modalities_present":video is not None 
            },
            "l":{
                "h1":h1_landmark, 
                "h2":h2_landmark,
                "embed":l_tok,
                "weighted":landmark_weighted,
                "modalities_present":landmarks is not None 
            }
        }

        mean_toks_inter = {
            "av":{
                "h1":h1_audio_video, 
                "h2":h2_audio_video,
                "embed_1":a_tok,
                "embed_2":v_tok,
                "weighted":audio_video_weighted,
                "modalities_present":mfcc is not None and video is not None
            },
            "va":{
                "h1":h1_video_audio, 
                "h2":h2_video_audio,
                "embed_1":v_tok,
                "embed_2":a_tok,
                "weighted":video_audio_weighted,
                "modalities_present":mfcc is not None and video is not None
            },
            "lv":{
                "h1":h1_landmark_video, 
                "h2":h2_landmark_video,
                "embed_1":l_tok,
                "embed_2":v_tok,
                "weighted":landmark_video_weighted,
                "modalities_present":landmarks is not None and video is not None
            },
        }
        # prepare empty losses dict
        losses: Dict[str, Dict[str, torch.Tensor]] = {
            "intra": {},  
            "inter": {},  
            "cls_loss":{}
        }
        # if not self.pretraining:
        # 1) intra‐modal SimCLR losses
        for name, m_tok in mean_toks_intra.items():
            tok_1 = m_tok["h1"]
            tok_2 = m_tok["h2"]
            weighted = m_tok["weighted"]
            embed = self._safe_mean(m_tok["embed"])
            if m_tok["modalities_present"]:
                if tok_1.numel() > 0 and tok_2.numel() > 0:
                    
                    context, mask = self.half_masking(embed)
                    predicted = self.predictor_e(context)
                    pred_loss_intra = F.mse_loss(predicted, embed)
                    losses["intra"][name] = 0.6 * self.contrastive(tok_1, tok_2, weighted=weighted) + 0.4 * pred_loss_intra
                    # losses["intra"][name] = self.contrastive(tok_1, tok_2, weighted=weighted)

        
        for name, m_tok in mean_toks_inter.items():
            tok_1 = m_tok["h1"]
            tok_2 = m_tok["h2"]
            weighted = m_tok["weighted"]
            embed_1 = self._safe_mean(m_tok["embed_1"])
            embed_2 = self._safe_mean(m_tok["embed_2"])
            
            if m_tok["modalities_present"]:
                if tok_1.numel() > 0 and tok_2.numel() > 0:
                    embed_1_p = self.predictor_e(embed_1)
                    pred_loss_inter = F.mse_loss(embed_2, embed_1_p)
                    losses["inter"][name] =  0.6 * self.contrastive(tok_1, tok_2,  weighted=weighted) + 0.4 * pred_loss_inter 

                # losses["inter"][name] = self.contrastive(tok_1, tok_2,  weighted=weighted) 

        

       
        if not self.pretraining:
            # logits_cls = logits_cls.clamp(min=-20.0, max=+20.0)
            losses["cls_loss"]["loss"] = self.cls(logits_cls.view(-1), labels)
       
        return logits_cls, losses, labels, v_tok_b


        import time

if __name__ == "__main__":
    import time

    model = HashFormer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Example inputs
    audio = torch.randn(1,16000).to(device)
    video = torch.randn(1,16,3,224,224).to(device)
    landmarks = torch.randn(1,478,2).to(device)
    video_aug=video
    audio_aug = audio
    # Warm-up
    for _ in range(5):
        with torch.no_grad():
            _ = model(audio=audio, video=video, video_aug=video, landmarks=landmarks)

    # Timing
    start = time.time()
    with torch.no_grad():
        _ = model(audio=audio, video=video, video_aug=video, landmarks=landmarks)
    end = time.time()

    print(f"Inference time: {(end - start)*1000:.2f} ms")

    # input_res = {
    #     "audio":     (16000,),            # → torch.empty((1,16000))
    #     "video":     (3, 16, 224, 224),   # → torch.empty((1,3,16,224,224))
    #     "video_aug": (3, 16, 224, 224),
    #     "landmarks": (478, 2),
    # }

    device = next(model.parameters()).device  # automatically detect model device

    def input_constructor(input_res):
        video_shape, audio_shape = input_res
        return {
            "video": torch.randn(2, *video_shape, device=device),
            "audio": torch.randn(2, *audio_shape, device=device),
            "video_aug": torch.randn(2, *video_shape, device=device),
            "audio_aug": torch.randn(2, *audio_shape, device=device)
        }

    input_res = ((16, 3, 224, 224), (16000,))
    macs, params = get_model_complexity_info(
        model,
        input_res,
        as_strings=True,
        print_per_layer_stat=False,
        input_constructor=input_constructor
    )
    
    print(f"FLOPs: {macs}, Params: {params}")
