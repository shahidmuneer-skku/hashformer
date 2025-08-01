from enum import Enum
import functools
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
import numpy as np
from beartype import beartype
from beartype.typing import Tuple, Optional, Union
# from timm.layers import get_3d_sincos_pos_embed
from torchaudio.transforms import Spectrogram
from torchvision.models import resnet50
# option A – xFormers (pip install xformers)
# from xformers.ops.fmha import rotary  # rotary.apply_rotary
# cos, sin = rotary.get_fixed_cos_sin(seq_len, head_dim, device)

# option B – flash-attn v2 (pip install flash-attn)
# from flash_attn.ops.rotary import apply_rotary_pos_emb as apply_rotary
# from flash_attn.ops.rotary import rotary_embedding  # builds cos/sin

# option C – write 6 lines yourself (shown below)

# constants

class TokenTypes(Enum):
    AUDIO = 0
    AUDIO_FUSION = 1
    VIDEO = 2
    VIDEO_FUSION = 3
    LANDMARK = 4
    LANDMARK_FUSION = 5
    GLOBAL = 7





# functions

def exists(val):
    return val is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor

def pair(t):
    return (t, t) if not isinstance(t, tuple) else t

def cum_mul(it):
    return functools.reduce(lambda x, y: x * y, it, 1)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# decorators

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# bias-less layernorm

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# geglu feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

class SELU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.selu(gate) * x
def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias = False)
    )

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# implemented from https://gist.github.com/srmsoumya/ce14ccdf50a3089a6ebe860789ae7dd3
# attention
def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: 3d tuple of grid size: t, h, w
    return:
    pos_embed: L, D
    """

    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def build_rope_cache(seq_len: int, dim: int, device):
    """Returns cos[seq_len, dim], sin[seq_len, dim] tensors."""
    theta = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device) / dim))
    pos   = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)                         # [seq, dim/2]
    emb   = torch.cat((freqs, freqs), dim=-1)               # repeat for even/odd
    cos, sin = emb.cos(), emb.sin()                         # [seq, dim]
    return cos, sin

class RopeCache(nn.Module):
    """
    Keeps cos/sin lookup tables and auto‑expands them when needed.
    Call   cos, sin = cache(seq_len, device, dtype)
    """
    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        self.dim  = dim
        self.base = base
        # start with a minimal table so first forward builds exactly what is needed
        self.register_buffer("_cos", torch.empty(0), persistent=False)
        self.register_buffer("_sin", torch.empty(0), persistent=False)

    def forward(self, seq_len: int, device, dtype):
        # if the cache is too small (or on a wrong device / dtype) ‑→ rebuild
        if (seq_len > self._cos.shape[0]         or
            self._cos.device != device           or
            self._cos.dtype  != dtype):
            self._build(seq_len, device, dtype)
        return self._cos[:seq_len], self._sin[:seq_len]

    @torch.no_grad()
    def _build(self, seq_len, device, dtype):
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=dtype) / self.dim))
        pos   = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(pos, theta)                # [seq, dim/2]
        emb   = torch.cat((freqs, freqs), dim=-1)      # [seq, dim]
        cos, sin = emb.cos(), emb.sin()
        self._cos = cos
        self._sin = sin

def apply_rotary(x, cos, sin):
    """
    x: [B, H, N, D]   (after you reshape with einops)
    cos/sin: [N, D]
    """
    cos = cos[None, None, :, :]         # broadcast over batch & heads
    sin = sin[None, None, :, :]
    x1, x2 = x[..., ::2], x[..., 1::2]  # even, odd
    x_rot  = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return x * cos + x_rot * sin


def get_positional_embeddings(seq_len, dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)
    embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
    return embeddings

def apply_RoPE(x, positional_embeddings):
    seq_len, dim = x.shape[1], x.shape[2]
    x_rotated = torch.einsum("bnd,nd->bnd", x, positional_embeddings)
    return x_rotated
    
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        dropout=0.2
        self.norm = LayerNorm(dim)

        # self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        

        self.to_q = nn.Sequential(
            nn.Linear(dim, inner_dim, bias = False),
            nn.Dropout(dropout)
        )
        self.to_kv = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias = False),
            nn.Dropout(dropout)
        )
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
          
           # Add bilinear transformation matrix
        self.W_bilinear = nn.Parameter(torch.randn(self.heads, dim_head, dim_head))
        # Initialize as identity for stability
        nn.init.eye_(self.W_bilinear.view(-1, dim_head))
        
        
        # self.attn_dropout = nn.Dropout(dropout)  # Dropout on attention scores if needed
        # NEW — one cache per Attention layer
        self.rope = RopeCache(dim_head, base=10_000)

    @staticmethod
    def _apply_rotary(x, cos, sin):
        """
        x : [B, H, N, D]      cos/sin : [N, D]
        """
        cos, sin = map(lambda t: t[None, None, :, :], (cos, sin))   # broadcast
        x1, x2   = x[..., ::2], x[..., 1::2]
        x_rot    = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return (x * cos) + (x_rot * sin)


    def forward(
        self,
        x,
        context = None,
        attn_mask = None
    ):
        x = self.norm(x)
        kv_x = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(kv_x).chunk(2, dim = -1))
         
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # # this method is opted from https://medium.com/%40DataDry/decoding-rotary-positional-embeddings-rope-the-secret-sauce-for-smarter-transformers-193cbc01e4ed
        seq_len = q.shape[2]
        device = x.device
       
        # This method is from chatgpt
        # ─── Rotary Positional Embeddings ──────────────────────
        # Rotary for query / key of *different* lengths
        seq_q, seq_k = q.shape[-2], k.shape[-2]
        max_len      = max(seq_q, seq_k)
        cos_base, sin_base = self.rope(max_len, device=x.device, dtype=x.dtype)

        q = self._apply_rotary(q, cos_base[:seq_q], sin_base[:seq_q])
        k = self._apply_rotary(k, cos_base[:seq_k], sin_base[:seq_k])
        # ─── Scaled dot‑product attention ─────────────────────
        
        
        q = q * self.scale

        k_transformed = torch.einsum('bhnd,hde->bhne', k, self.W_bilinear)
        sim = einsum('b h i d, b h j d -> b h i j', q, k_transformed)

        # sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# main class

class MaskEncoder(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        num_fusion_tokens = 16,
        audio_patch_size: Union[int, Tuple[int, int]] = 16,
        video_patch_size: Union[int, Tuple[int, int]] = 16,
        video_temporal_patch_size = 5,
        video_channels = 3,
        spec_n_fft = 128,
        spec_power = 2,
        spec_win_length = 24,
        spec_hop_length = None,
        spec_pad = 0,
        spec_center = True,
        spec_pad_mode = 'reflect',
        spec_aug_stretch_factor = 0.8,
        spec_aug_freq_mask = 80,
        spec_aug_time_mask = 80,
        return_token_types: Tuple[TokenTypes] = (TokenTypes.AUDIO_FUSION, TokenTypes.AUDIO, TokenTypes.VIDEO_FUSION, TokenTypes.VIDEO,TokenTypes.LANDMARK_FUSION,TokenTypes.LANDMARK),
        max_pos_video: int = 5880,
        max_pos_landmarks: int = 1024,
        max_pos_audio: int = 2048,
    ):
   

    #  
        super().__init__()
        self.max_return_tokens = len(return_token_types)
        
        self.hash_bit = 64
        self.fc = nn.Linear(512, 512)
        self.b = nn.Linear(512, self.hash_bit)

        self.return_token_types = return_token_types
        return_token_types_tensor = torch.tensor(list(map(lambda t: t.value, return_token_types)))
        self.register_buffer('return_token_types_tensor', return_token_types_tensor, persistent = False)

        self.return_tokens = nn.Parameter(torch.randn(self.max_return_tokens, dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)
        self.pos_embeds = nn.ModuleDict({
                    "audio": nn.Embedding(512, dim),
                    "video": nn.Embedding(512, dim),
                })
        self.dim = dim
        self.max_pos = {
            "audio": max_pos_audio,
            "video": max_pos_video,
            "landmarks": max_pos_landmarks
        }
        # audio input

        self.audio_patch_size = audio_patch_height, audio_patch_width = pair(audio_patch_size)

        self.spec = Spectrogram(
            n_fft = spec_n_fft,
            power = spec_power,
            win_length = spec_win_length,
            hop_length = spec_hop_length,
            pad = spec_pad,
            center = spec_center,
            pad_mode = spec_pad_mode
        )

        audio_input_dim = cum_mul(self.audio_patch_size)
        # print(audio_patch_size)
        # self.audio_to_tokens = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=40,
        #         out_channels=audio_input_dim,
        #         kernel_size=audio_patch_size,
        #         stride=audio_patch_size
        #     ),
        #     Rearrange("b d t -> b t d"),
        #     nn.LayerNorm(audio_input_dim),
        #     nn.Linear(audio_input_dim, dim),
        #     nn.LayerNorm(dim)
        # )

        audio_input_dim = cum_mul(self.audio_patch_size)
        self.audio_to_tokens = nn.Sequential(
            Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1 = audio_patch_height, p2 = audio_patch_width),
            nn.LayerNorm(audio_input_dim),
            nn.Linear(audio_input_dim, dim),
            nn.LayerNorm(dim)
        )

        # video input

        self.video_patch_size = (video_temporal_patch_size, *pair(video_patch_size))

        video_input_dim = cum_mul(self.video_patch_size) * video_channels
        video_patch_time, video_patch_height, video_patch_width = self.video_patch_size
      
        # video_grid = (18, 14, 14)           # pick an upper bound
        video_patch_time, video_patch_height, video_patch_width = self.video_patch_size
        self.video_grid = (
            max_pos_video // video_patch_time,          # temporal grid
            224 // video_patch_height,                  # height grid — or video.shape[-2] if dynamic
            224 // video_patch_width                    # width grid — or video.shape[-1] if dynamic
        )
        pos_3d = get_3d_sincos_pos_embed(
            embed_dim = dim,
            grid_size = self.video_grid,
            cls_token = False
          )   # returns [T*H*W, dim] numpy\
    

    
    
        self.register_buffer("video_pos_table",
            torch.from_numpy(pos_3d).float(), persistent=False)
        
        self.video_to_tokens = nn.Sequential(
            Rearrange('b c (t p1) (h p2) (w p3) -> b t h w (c p1 p2 p3)', p1 = video_patch_time, p2 = video_patch_height, p3 = video_patch_width),
            nn.LayerNorm(video_input_dim),
            nn.Linear(video_input_dim, dim),
            nn.LayerNorm(dim)
        )
       

        

        # fusion tokens

        self.landmarks_fusion_tokens = nn.Parameter(torch.randn(num_fusion_tokens, dim))
        self.video_fusion_tokens = nn.Parameter(torch.randn(num_fusion_tokens, dim))
        self.audio_fusion_tokens = nn.Parameter(torch.randn(num_fusion_tokens, dim))

        # transformer

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))
        
        self.norm = LayerNorm(dim)
        self.fc_norm = nn.LayerNorm(512, eps=1e-6)
        self.fc = nn.Linear(512, 512)
        self.b = nn.Linear(512, self.hash_bit)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc.weight, mode='fan_out')
        nn.init.normal_(self.fc.bias)
        nn.init.kaiming_uniform_(self.b.weight, mode='fan_out')
        nn.init.normal_(self.b.bias)

    # ---------------- private helpers ----------------
    def _ensure_pos_embed(self, modality: str, length: int):
        """Extend positional table on‑the‑fly if sequence is longer than current."""
        table: nn.Embedding = self.pos_embeds[modality]
        if length <= table.num_embeddings:
            return table
        new_len = min(self.max_pos[modality], max(length, table.num_embeddings * 2))
        if new_len <= table.num_embeddings:
            raise ValueError(f"{modality} sequence length {length} exceeds max_pos_{modality}={self.max_pos[modality]}")
        # print(new_len, self.dim)
        new_table = nn.Embedding(new_len, self.dim, device=table.weight.device)
        new_table.weight.data[: table.num_embeddings] = table.weight.data
        nn.init.normal_(new_table.weight.data[table.num_embeddings:])
        self.pos_embeds[modality] = new_table
        return new_table

    def hash_head(self, x):
        x = F.relu(self.fc(x))
        x = torch.tanh(self.b(x))
        h = x.sign()
        return x + torch.detach(h - x)

    def cross_attention(self, x1, x2, T=1):
        sim = torch.einsum('nid,njd->nij', F.normalize(x1, dim=-1), F.normalize(x2, dim=-1))
        sim1 = torch.softmax(sim.permute(0, 2, 1) / T, dim=-1)
        sim2 = torch.softmax(sim / T, dim=-1)
        x1 = torch.einsum('nid,ndj->nij', sim1, x1)
        x2 = torch.einsum('nid,ndj->nij', sim2, x2)
        return x1, x2

    def weighted(self, x1, x2, T=0.2):
        sim = torch.einsum('nid,mjd->nmij', F.normalize(x1, dim=-1), F.normalize(x2, dim=-1))
        sim = sim.max(-1)[0].mean(-1)
        sim = torch.softmax(sim / T, dim=-1)
        scale = torch.eye(sim.size(0), device=sim.device)
        scale = torch.einsum('ni,ni->n', scale, sim)
        weighted = (sim / scale)
        return weighted

    def forward(
        self,
        *,
        audio,
        video,
        landmarks, 
        return_token_indices: Optional[Tuple[int]] = None
    ):
        # batch, device = audio.shape[0], audio.device
    
        # validate video can be patched
      
        if video is not None:
            assert all([divisible_by(numer, denom) for denom, numer in zip(self.video_patch_size, tuple(video.shape[-3:]))]), f'video shape {video.shape[-3:]} needs to be divisible by {self.video_patch_size}'



        batch_size = audio.shape[0] if audio is not None else (
            video.shape[0] if video is not None else (
                landmarks.shape[0] if landmarks is not None else (
                    text_tokens.shape[0] if text_tokens is not None else 1
                )
            )
        ) 
        device = audio.device if audio is not None else (
            video.device if video is not None else (
                landmarks.device if landmarks is not None else (
                    text_tokens.device if text_tokens is not None else 1
                )
            )
        )
        if audio is not None:
            audio = self.spec(audio)
            
            height, width = audio.shape[-2:]
            patch_height, patch_width = self.audio_patch_size
            rounded_height, rounded_width = map(lambda args: round_down_nearest_multiple(*args), ((height, patch_height), (width, patch_width)))

            if (height, width) != (rounded_height, rounded_width): # just keep printing to be annoying until it is fixed
                print_once(f'spectrogram yielded shape of {(height, width)}, but had to be cropped to {(rounded_height, rounded_width)} to be patchified for transformer')
                
            audio = audio[..., :rounded_height, :rounded_width]
            audio_tokens = self.audio_to_tokens(audio)

            # audio_tokens = self.audio_to_tokens(audio)
        else:
            audio_tokens = torch.zeros(batch_size, 1, self.dim, device=device)

        if video is not None:
            video_tokens = self.video_to_tokens(video)
        else:
            video_tokens = torch.zeros(batch_size, 1, self.dim, device=device)
        
        if landmarks is not None:
            landmarks_tokens = landmarks
        else:
            landmarks_tokens = torch.zeros(batch_size, 1, self.dim, device=device)


        video_fusion_tokens = repeat(self.video_fusion_tokens, 'n d -> b n d', b = batch_size)
        audio_fusion_tokens = repeat(self.audio_fusion_tokens, 'n d -> b n d', b = batch_size)
        landmarks_fusion_tokens = repeat(self.landmarks_fusion_tokens, 'n d -> b n d', b = batch_size)
        # print(audio_tokens.shape,video_tokens.shape)
        # construct all tokens
        audio_fusion_tokens, audio_tokens, video_fusion_tokens,video_tokens, landmarks_fusion_tokens, landmarks_tokens = map(lambda t: rearrange(t, 'b ... d -> b (...) d'), (audio_fusion_tokens, audio_tokens, video_fusion_tokens,video_tokens, landmarks_fusion_tokens, landmarks_tokens))
        

        # pos_table = self._ensure_pos_embed("audio", audio_tokens.size(1))
        # pos = pos_table.weight[: audio_tokens.size(1)][None]
        # audio_tokens = audio_tokens + pos
        
        v_pos = self.video_pos_table[:video_tokens.size(1)]
        
        video_tokens = video_tokens + v_pos[None]

        # print(
        #     audio_fusion_tokens.shape,
        #     audio_tokens.shape,
        #     video_fusion_tokens.shape,
        #     video_tokens.shape,
        #     landmarks_fusion_tokens.shape,
        #     landmarks_tokens.shape)
        tokens, ps = pack((
            audio_fusion_tokens,
            audio_tokens,
            video_fusion_tokens,
            video_tokens,
            landmarks_fusion_tokens,
            landmarks_tokens
        ), 'b * d')


        token_types = torch.tensor(list((
            *((TokenTypes.AUDIO_FUSION.value,) * audio_fusion_tokens.shape[-2]),
            *((TokenTypes.AUDIO.value,) * audio_tokens.shape[-2]),
            *((TokenTypes.VIDEO_FUSION.value,) * video_fusion_tokens.shape[-2]),
            *((TokenTypes.VIDEO.value,) * video_tokens.shape[-2]),
            *((TokenTypes.LANDMARK_FUSION.value,) * landmarks_fusion_tokens.shape[-2]),
            *((TokenTypes.LANDMARK.value,) * landmarks_tokens.shape[-2]),
        )), device = device, dtype = torch.long)

        token_types_attend_from = rearrange(token_types, 'i -> i 1')
        token_types_attend_to = rearrange(token_types, 'j -> 1 j')


        hash_mask = token_types_attend_from == token_types_attend_to

        # fusion can attend to everything

        hash_mask = hash_mask | (token_types_attend_from == TokenTypes.VIDEO_FUSION.value)
        hash_mask = hash_mask | (token_types_attend_from == TokenTypes.AUDIO_FUSION.value)
        hash_mask = hash_mask | (token_types_attend_from == TokenTypes.LANDMARK_FUSION.value)

        # attend and feedforward

        for attn, ff in self.layers:
            tokens = attn(tokens, attn_mask = hash_mask) + tokens
            tokens = ff(tokens) + tokens

        tokens = self.norm(tokens)
        af_tok, a_tok, vf_tok,v_tok, lf_tok, l_tok= unpack(tokens, ps, "b * d")
        # final attention pooling - each modality pool token can only attend to its own tokens
       
        audio_tokens_cross_a1, video_tokens_cross_a2 = self.cross_attention(a_tok, v_tok)
        audio_video_weighted = self.weighted(audio_tokens_cross_a1, video_tokens_cross_a2)


        a_tok_h1 = self.fc_norm(audio_tokens_cross_a1.mean(dim=1))
        a_tok_h2 = self.fc_norm(video_tokens_cross_a2.mean(dim=1))

        h1_audio_video = self.hash_head(a_tok_h1)
        h2_audio_video = self.hash_head(a_tok_h2)


        
        audio_tokens_cross_a1, audio_tokens_cross_a2 = self.cross_attention(a_tok, a_tok)
        audio_weighted = self.weighted(audio_tokens_cross_a1, audio_tokens_cross_a2)


        a_tok_h1 = self.fc_norm(audio_tokens_cross_a1.mean(dim=1))
        a_tok_h2 = self.fc_norm(audio_tokens_cross_a2.mean(dim=1))

        h1_audio = self.hash_head(a_tok_h1)
        h2_audio = self.hash_head(a_tok_h2)


        video_audio_tokens_cross_v1, video_audio_tokens_cross_v2 = self.cross_attention(v_tok, a_tok)
        video_audio_weighted = self.weighted(video_audio_tokens_cross_v1, video_audio_tokens_cross_v2)

        
        va_tok_h1 = self.fc_norm(video_audio_tokens_cross_v1.mean(dim=1))
        va_tok_h2 = self.fc_norm(video_audio_tokens_cross_v2.mean(dim=1))

        h1_video_audio = self.hash_head(va_tok_h1)
        h2_video_audio = self.hash_head(va_tok_h2)




        video_tokens_cross_v1, video_tokens_cross_v2 = self.cross_attention(v_tok, v_tok)
        video_weighted = self.weighted(video_tokens_cross_v1, video_tokens_cross_v2)

        v_tok_h1 = self.fc_norm(video_tokens_cross_v1.mean(dim=1))
        v_tok_h2 = self.fc_norm(video_tokens_cross_v2.mean(dim=1))

        h1_video = self.hash_head(v_tok_h1)
        h2_video = self.hash_head(v_tok_h2)

        

        landmark_video_tokens_cross_v1, landmark_video_tokens_cross_v2 = self.cross_attention(l_tok, v_tok)
        landmark_video_weighted = self.weighted(landmark_video_tokens_cross_v1, landmark_video_tokens_cross_v2)

        lv_tok_h1 = self.fc_norm(landmark_video_tokens_cross_v1.mean(dim=1))
        lv_tok_h2 = self.fc_norm(landmark_video_tokens_cross_v2.mean(dim=1))

        h1_landmark_video = self.hash_head(lv_tok_h1)
        h2_landmark_video = self.hash_head(lv_tok_h2)



        
        landmark_tokens_cross_l1, landmark_tokens_cross_l2 = self.cross_attention(l_tok, l_tok)
        landmark_weighted = self.weighted(landmark_tokens_cross_l1, landmark_tokens_cross_l2)


        l_tok_h1 = self.fc_norm(landmark_tokens_cross_l1.mean(dim=1))
        l_tok_h2 = self.fc_norm(landmark_tokens_cross_l2.mean(dim=1))

        h1_landmark = self.hash_head(l_tok_h1)
        h2_landmark = self.hash_head(l_tok_h2)
        # hashes = h1_landmark.squeeze()  # shape [B, bit]
        # bit_balance = hashes.mean(dim=0)
        # print("Per-bit mean:", bit_balance)
        # print("Overall sparsity:", bit_balance.mean())
        

        logits = (audio_weighted, h1_audio, h2_audio, audio_video_weighted, h1_audio_video, h2_audio_video,
                 video_weighted, h1_video, h2_video, video_audio_weighted, h1_video_audio, h2_video_audio, 
                 landmark_weighted, h1_landmark, h2_landmark, landmark_video_weighted, h1_landmark_video, h2_landmark_video)


        return_tokens = self.return_tokens
        return_token_types_tensor = self.return_token_types_tensor

        if exists(return_token_indices):
            assert len(set(return_token_indices)) == len(return_token_indices), 'all indices must be unique'
            assert all([indice < self.max_return_tokens for indice in return_token_indices]), 'indices must range from 0 to max_num_return_tokens - 1'

            return_token_indices = torch.tensor(return_token_indices, dtype = torch.long, device = device)

            return_token_types_tensor = return_token_types_tensor[return_token_indices]
            return_tokens = return_tokens[return_token_indices]

        return_tokens = repeat(return_tokens, 'n d -> b n d', b = batch_size)
        pool_mask = rearrange(return_token_types_tensor, 'i -> i 1') == token_types_attend_to
        # global queries can attend to all tokens
        pool_mask = pool_mask | rearrange(return_token_types_tensor, 'i -> i 1') == torch.ones_like(token_types_attend_to, dtype=torch.long) * TokenTypes.GLOBAL.value

        pooled_tokens = self.attn_pool(return_tokens, context = tokens, attn_mask = pool_mask) + return_tokens
        return pooled_tokens,af_tok, a_tok,vf_tok,v_tok, lf_tok, l_tok, logits