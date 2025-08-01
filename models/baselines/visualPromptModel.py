# -*- coding: utf-8 -*-
"""
Multi‑modal Deepfake Detector with Multi‑task Audio‑Visual Prompt Learning
=======================================================================
Reference implementation of:
  "Multi‑modal Deepfake Detection via Multi‑task Audio‑Visual Prompt Learning" – AAAI‑25.

Highlights
----------
* Frozen back‑bones:
    • CLIP ViT‑B/32 (image branch)
    • Whisper‑base encoder (audio branch)
* Parameter‑efficient prompt adapters:
    • Sequential Visual Prompts (one learnable token per ViT layer)
    • Short‑Time Audio Prompts (concatenate & project at each Whisper layer)
* Heads:
    • Visual authenticity head (2‑layer MLP)
    • Audio authenticity head  (2‑layer MLP)
    • Fused head              (2‑layer MLP)
* Frame‑Level Cross‑Modal Feature‑Matching (CMFM) loss

The model expects **raw RGB frames** (B×T×3×H×W, 0‑1) and **raw waveforms**
(B×samples, 16 kHz).  Pre‑processing (patchifying frames, log‑mel extraction)
relies on CLIPFeatureExtractor and WhisperFeatureExtractor from 🤗
`transformers`.  For brevity, only the core network & CMFM loss are defined –
connect this module to your dataloader / training loop.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPModel,
    CLIPImageProcessor,
    WhisperModel,
    WhisperFeatureExtractor,
)

# -----------------------------------------------------------------------------
# Prompt modules
# -----------------------------------------------------------------------------
class SequentialVisualPrompt(nn.Module):
    """One learnable prompt token injected *per* ViT layer (prepended).

    Notes
    -----
    * We keep previously injected prompts when moving to deeper layers –
      hence the term *sequential*.
    * N_vpt (num_tokens) = 1 by default, following paper.
    """

    def __init__(self, n_layers: int, dim: int, num_tokens: int = 1):
        super().__init__()
        self.num_tokens = num_tokens
        self.prompts = nn.ParameterList(
            [nn.Parameter(torch.randn(num_tokens, dim) * 0.02) for _ in range(n_layers)]
        )

    def forward(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """prepend prompt tokens for *layer_idx*.

        Parameters
        ----------
        hidden : (B, L, D)
        layer_idx : int  (0‑based index of transformer layer)
        """
        B = hidden.size(0)
        p = self.prompts[layer_idx].unsqueeze(0).expand(B, -1, -1)
        return torch.cat([p, hidden], dim=1)


class ShortTimeAudioPrompt(nn.Module):
    def __init__(self, n_layers: int, in_dim: int):
        super().__init__()
        # one prompt vector per layer
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(in_dim) * 0.02)
            for _ in range(n_layers)
        ])
        self.proj = nn.Linear(in_dim * 2, in_dim)
        self.global_token = nn.Parameter(torch.zeros(1, 1, in_dim))
        nn.init.trunc_normal_(self.global_token, std=0.02)

    def forward(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # hidden: (B, L, D)
        B, L, D = hidden.size()
        # get the single prompt vector and expand across time
        p = self.prompts[layer_idx]              # (D,)
        p = p.unsqueeze(0).unsqueeze(0)          # (1, 1, D)
        p = p.expand(B, L, D)                    # (B, L, D)
        cat = torch.cat([hidden, p], dim=-1)     # (B, L, 2D)
        cat = self.proj(cat)                     # back to (B, L, D)
        # prepend global token only at layer 0
        if layer_idx == 0:
            g = self.global_token.expand(B, -1, -1)  # (B, 1, D)
            cat = torch.cat([g, cat], dim=1)         # (B, L+1, D)
        return cat

# -----------------------------------------------------------------------------
# Cross‑Modal Feature Matching Loss
# -----------------------------------------------------------------------------
class CMFMLoss(nn.Module):
    """Frame‑Level Cross‑Modal Feature Matching Loss.

    Encourages low cosine distance for aligned real pairs, high distance for
    (1) fake pairs within same sample, (2) cross‑segment pairs.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 2.0, gamma: float = 1.0):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    @staticmethod
    def _cos(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - F.cosine_similarity(x, y, dim=-1)  # distance in [0,2]

    def forward(
        self,
        f_v: torch.Tensor,  # (B, T, D)
        f_a: torch.Tensor,  # (B, T, D)
        labels: torch.Tensor,  # (B,) 0=real,1=fake
    ) -> torch.Tensor:
        B, T, _ = f_v.shape
        loss_pos, cnt_pos = 0.0, 0
        loss_neg, cnt_neg = 0.0, 0

        # same‑segment pairs
        for b in range(B):
            for t in range(T):
                d = self._cos(f_v[b, t], f_a[b, t])
                if labels[b] == 0:  # real → pull together
                    loss_pos += d * self.alpha
                    cnt_pos += 1
                else:  # fake → push apart
                    loss_neg += (1 - d) * self.beta  # we want cosine distance high
                    cnt_neg += 1

        # cross‑segment pairs (mismatch) – push apart
        for i in range(B):
            for j in range(B):
                if i == j:
                    continue
                d = self._cos(f_v[i].view(-1, f_v.size(-1)), f_a[j].view(-1, f_a.size(-1))).mean()
                loss_neg += (1 - d) * self.gamma
                cnt_neg += 1

        loss = 0.0
        if cnt_pos:
            loss += loss_pos / cnt_pos
        if cnt_neg:
            loss += loss_neg / cnt_neg
        return loss


# -----------------------------------------------------------------------------
# Main Model – Multi‑task Audio‑Visual Prompt Network
# -----------------------------------------------------------------------------
class MultiModalPromptDetector(nn.Module):
    """AAAI‑25 Multi‑task Audio‑Visual Prompt Deepfake Detector.

    Inputs
    ------
    frames : torch.Tensor
        Shape (B, T, 3, H, W) RGB 0‑1.
    audio  : torch.Tensor
        Shape (B, samples) 16 kHz mono waveform.
    """

    def __init__(
        self,
        visual_backbone: str = "openai/clip-vit-base-patch32",
        audio_backbone: str = "openai/whisper-base",
        num_vprompt_tokens: int = 1,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        # ------------------------------------------------------------------
        # Vision Branch
        # ------------------------------------------------------------------
        self.clip = CLIPModel.from_pretrained(visual_backbone).vision_model
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip_processor = CLIPImageProcessor.from_pretrained(visual_backbone)

        n_v_layers = len(self.clip.encoder.layers)
        d_v = self.clip.config.hidden_size
        self.v_prompts = SequentialVisualPrompt(n_v_layers, d_v, num_vprompt_tokens)
        self.v_head = nn.Sequential(nn.Linear(d_v, 256), nn.ReLU(), nn.Linear(256, 1))

        # ------------------------------------------------------------------
        # Audio Branch
        # ------------------------------------------------------------------
        self.whisper = WhisperModel.from_pretrained(audio_backbone).encoder
        for p in self.whisper.parameters():
            p.requires_grad = False
        self.whisper_processor = WhisperFeatureExtractor()
        n_a_layers = len(self.whisper.layers)
        d_a = self.whisper.config.d_model
        # Prompt sequence length = 1 (short‑time) – matches paper Figure 2
        self.a_prompts = ShortTimeAudioPrompt(n_a_layers, in_dim=d_a)
        self.a_head = nn.Sequential(nn.Linear(d_a, 256), nn.ReLU(), nn.Linear(256, 1))

        # ------------------------------------------------------------------
        # Fusion + Align
        # ------------------------------------------------------------------
        self.align_v = nn.Conv1d(d_v, 256, kernel_size=1)
        self.align_a = nn.Conv1d(d_a, 256, kernel_size=1)
        self.f_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

        # Loss
        self.cmfm_loss = CMFMLoss()
        self.bce = nn.BCELoss()

        self.device = device
        self.to(device)

    # ------------------------------------------------------------------
    # Utility – preprocessing
    # ------------------------------------------------------------------
    def _preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Convert raw frames to patch embeddings expected by CLIP ViT.

        Returns (B*T, L, D).
        """
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        flat = torch.flatten(frames,start_dim=0, end_dim=1).to(self.device)
        with torch.no_grad():
            pixel_values = (
                self.clip_processor(images=flat, return_tensors="pt")
                .pixel_values
                .to(self.device)
            )
            hidden = self.clip.embeddings(pixel_values)
        return hidden, B, T

    
    def _preprocess_audio(self, audio: torch.Tensor, n_frames: int):
        """Extract initial Whisper embeddings with proper positional encodings.

        Handles both legacy (`positional_embedding`) and current (`embed_positions` or
        `get_position_embeddings`) attribute names.
        """
        with torch.no_grad():
            feats = self.whisper_processor(audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
            log_mel = feats.input_features.to(audio.device)
            hs = self.whisper.conv1(log_mel)
            hs = F.gelu(hs)
            if hasattr(self.whisper, "conv2"):
                hs = self.whisper.conv2(hs)
            hs = hs.transpose(1, 2)  # (B, T', D)
            L = hs.size(1)
            # Add positional embeddings
            if hasattr(self.whisper, "embed_positions"):
                pos_ids = torch.arange(L, device=audio.device, dtype=torch.long)[None, :]
                pos = self.whisper.embed_positions(pos_ids)

            elif hasattr(self.whisper, "get_position_embeddings"):
                pos = self.whisper.get_position_embeddings(hs.size(1)).unsqueeze(0).expand(hs.size(0), -1, -1)
            else:
                raise AttributeError("No positional embedding accessor found in Whisper encoder.")
            hs = hs + pos
        return hs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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

    ):
    

        # -------- Vision branch --------
        v_hidden, B, T = self._preprocess_frames(video)
        # iterate through ViT layers with prompts
        for idx, layer in enumerate(self.clip.encoder.layers):
            v_hidden = self.v_prompts(v_hidden, idx)
            v_hidden = layer(v_hidden, attention_mask=None, causal_attention_mask=None)[0]
        # take first prompt token (index 1) as frame representation
        v_prompt_tok = v_hidden[:, 0, :]  # (B*T, D_v)
        v_prompt_tok = v_prompt_tok.view(B, T, -1)  # (B, T, D_v)
        v_segment = v_prompt_tok.mean(dim=1)  # (B, D_v)
        v_score = torch.sigmoid(self.v_head(v_segment))  # (B,1)

        # -------- Audio branch --------
        a_hidden = self._preprocess_audio(audio, T)  # (B, L, D_a)
        for idx, layer in enumerate(self.whisper.layers):
            a_hidden = self.a_prompts(a_hidden, idx)
            a_hidden = layer(a_hidden,attention_mask=None, layer_head_mask=None )[0]
        # discard first token (global) for segment feature
        a_feat_no_global = a_hidden[:, 1:, :]
        a_prompt_feat = a_feat_no_global.mean(dim=1)  # (B, D_a)
        a_score = torch.sigmoid(self.a_head(a_prompt_feat))

        # -------- Fusion --------
        v_align = self.align_v(v_prompt_tok.transpose(1, 2))  # (B, 256, T)
        a_align = self.align_a(a_hidden[:, 1:, :].transpose(1, 2))  # match T via conv stride=1
        # simple temporal average pooling
        fused = torch.cat([v_align.mean(dim=-1), a_align.mean(dim=-1)], dim=1)  # (B,512)
        f_score = torch.sigmoid(self.f_head(fused))

        if labels is None:
            return {
                "v_score": v_score.squeeze(1),
                "a_score": a_score.squeeze(1),
                "f_score": f_score.squeeze(1),
            }

        # -------- Losses --------
        labels = labels.float().to(self.device)
        v_score = v_score.to(self.device)
        a_score = a_score.to(self.device)
        f_score = f_score.to(self.device)
        
        lv = self.bce(v_score.squeeze(1), labels)
        la = self.bce(a_score.squeeze(1), labels)
        lf = self.bce(f_score.squeeze(1), labels)

        if a_align.size(2) != T:
            a_align = F.interpolate(
                a_align,
                size=T,
                mode="linear",
                align_corners=False
            )
        
        v_feat = v_align.transpose(1, 2)
        a_feat = a_align.transpose(1, 2)
        
        # lcmfm = self.cmfm_loss(v_prompt_tok, a_hidden[:, 1:, :].view(B, T, -1), labels)
        lcmfm = self.cmfm_loss(v_feat, a_feat, labels)
        loss = lv + la + lf + lcmfm
        return {
            "loss": loss,
            "l_v": lv.detach(),
            "l_a": la.detach(),
            "l_f": lf.detach(),
            "l_cmfm": lcmfm.detach(),
            "v_score": v_score.squeeze(1).detach(),
            "a_score": a_score.squeeze(1).detach(),
            "f_score": f_score.squeeze(1).detach(),
        }


# -----------------------------------------------------------------------------
# Quick sanity check – dummy forward pass
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    B, T, H, W = 2, 4, 224, 224
    frames = torch.rand(B, T, 3, H, W)
    wav = torch.randn(B, 16000)  # 1‑second audio
    labels = torch.randint(0, 2, (B,)).float()

    model = MultiModalPromptDetector(device="cpu")
    out = model(frames, wav, labels)
    print("loss", out["loss"].item())
