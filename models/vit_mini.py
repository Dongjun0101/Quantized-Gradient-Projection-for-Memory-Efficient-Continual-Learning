import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import os
import os.path
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
import numpy as np
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import pickle
import numpy as np
import pickle
import os
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms
from collections import OrderedDict
import torch.nn.functional as F


import torch
import torch.nn as nn
from collections import OrderedDict
import math

import re
from typing import Union, Dict

def _to_state_dict(obj) -> Dict[str, torch.Tensor]:
    """Accept a path, a dict-like checkpoint, a timm/torch model, or raw state_dict; return clean state_dict."""
    if isinstance(obj, str):
        ckpt = torch.load(obj, map_location='cpu')
    elif hasattr(obj, "state_dict"):
        ckpt = obj.state_dict()
    else:
        ckpt = obj

    if not isinstance(ckpt, dict):
        raise TypeError("Unsupported checkpoint type.")

    # unwrap common containers
    for key in ['state_dict', 'model', 'ema', 'net']:
        if key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]

    # strip 'module.' prefix (DDP) if present
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    return ckpt

def _find_pos_embed_key(sd: Dict[str, torch.Tensor]):
    # timm uses 'pos_embed', some others 'positional_embedding'
    for k in ['pos_embed', 'positional_embedding', 'encoder.pos_embedding']:
        if k in sd:
            return k
    # Sometimes nested like 'model.pos_embed' (rare). Fallback: last key that endswith 'pos_embed'
    cands = [k for k in sd.keys() if k.endswith('pos_embed')]
    return cands[-1] if cands else None

def _num_blocks_from_state_dict(sd: Dict[str, torch.Tensor]) -> int:
    pat = re.compile(r'^blocks\.(\d+)\.')
    max_idx = -1
    for k in sd.keys():
        m = pat.match(k)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1

@torch.no_grad()
def _interpolate_pos_embed(pos_src: torch.Tensor, dst_num_patches: int, dst_extra_tokens: int) -> torch.Tensor:
    """
    pos_src: (1, src_tokens, C) with 1 or 2 extra tokens at front (cls[, dist])
    Returns: (1, dst_extra_tokens + dst_num_patches, C) with bicubic-resized patch grid.
    """
    assert pos_src.ndim == 3 and pos_src.shape[0] == 1
    C = pos_src.shape[-1]

    # Heuristically guess number of extra tokens at front (1 for cls, or 2 for cls+dist)
    # Try 2, then 1, then clamp to 1 if not square.
    for src_extra_tokens in (2, 1):
        num_patches_src = pos_src.shape[1] - src_extra_tokens
        g = int(round(num_patches_src ** 0.5))
        if g * g == num_patches_src and num_patches_src > 0:
            break
    else:
        # fallback assume 1
        src_extra_tokens = 1
        num_patches_src = pos_src.shape[1] - src_extra_tokens
        g = int(round(num_patches_src ** 0.5))

    extra_src = pos_src[:, :src_extra_tokens]
    patch_pos_src = pos_src[:, src_extra_tokens:]  # (1, N_src, C)

    gs_src = int(math.sqrt(patch_pos_src.shape[1]))
    gs_dst = int(math.sqrt(dst_num_patches))
    assert gs_src * gs_src == patch_pos_src.shape[1], "Source pos_embed patch count is not a square."
    assert gs_dst * gs_dst == dst_num_patches, "Destination pos_embed patch count is not a square."

    patch_pos_src = patch_pos_src.reshape(1, gs_src, gs_src, C).permute(0, 3, 1, 2)  # (1,C,gs_src,gs_src)
    patch_pos_dst = F.interpolate(patch_pos_src, size=(gs_dst, gs_dst), mode='bicubic', align_corners=False)
    patch_pos_dst = patch_pos_dst.permute(0, 2, 3, 1).reshape(1, gs_dst * gs_dst, C)

    # If dst has fewer extra tokens than src, keep only the first (cls)
    if dst_extra_tokens == 1:
        extra_dst = extra_src[:, :1, :]
    else:
        # rarely used in this project; keep up to dst_extra_tokens
        extra_dst = extra_src[:, :dst_extra_tokens, :]

    return torch.cat([extra_dst, patch_pos_dst], dim=1)





def count_sketch_np(mat: np.ndarray, C: int, seed: int = 0) -> np.ndarray:
    """
    Perform Count Sketch on 'mat' (shape (D, N)) to compress from N -> C columns.
    Return shape (D, C).
    Args:
      mat: np.ndarray of shape (D, N)
      C:   int, # of buckets (e.g., set to D if you want (D, D))
      seed: random seed for reproducibility
    """
    rng = np.random.RandomState(seed)

    D, N = mat.shape
    
    # # 1) random hash function h(i) in [0..C-1]
    # h = rng.randint(low=0, high=C, size=N)  # shape (N,)
    # # 2) random sign function s(i) in {+1, -1}
    # signs = rng.choice([-1, +1], size=N)

    # # 3) Initialize output
    # out = np.zeros((D, C), dtype=mat.dtype)

    # # 4) accumulate
    # # for i in range(N):
    # #     bucket = h[i]
    # #     out[:, bucket] += signs[i] * mat[:, i]
    # np.add.at(out, (slice(None), h), signs * mat)

    # 1) Create sparse random matrix S of shape (N, C)
    # Each column has exactly one non-zero entry (either +1 or -1)
    h = rng.randint(low=0, high=C, size=N)  # hash function: (N,)
    signs = rng.choice([-1, +1], size=N)     # signs: (N,)
    
    # Create sparse matrix S: (N, C)
    # S[i, h[i]] = signs[i] for all i
    S = np.zeros((N, C), dtype=mat.dtype)
    S[np.arange(N), h] = signs
    
    # 2) Matrix multiplication: mat @ S
    # (D, N) @ (N, C) = (D, C)
    out = mat @ S

    return out

def multi_head_attention_custom(
    x_input: torch.Tensor,
    attn_layer: torch.nn.MultiheadAttention,
    *,
    attn_mask: torch.Tensor = None,
    key_padding_mask: torch.Tensor = None,
    need_weights: bool = False,
    average_attn_weights: bool = True,
    training: bool = True,
    dropout_p: float = 0.0,
    ):
    """
    A re-implementation of scaled-dot-product multi-head self-attention
    that includes some advanced PyTorch features:
      - attention masks
      - key padding mask
      - dropout on attention weights
      - returning optional attn_weights
    """

    # x_input shape = (B, seq_len, D)
    B, seq_len, D = x_input.shape
    num_heads = attn_layer.num_heads   # e.g. 4
    head_dim  = D // num_heads         # dimension per head

    # ----- 1) PROJECT Q, K, V from x_input -----
    #        in_proj_weight: (3*D, D)
    W_in  = attn_layer.in_proj_weight
    b_in  = attn_layer.in_proj_bias
    W_out = attn_layer.out_proj.weight
    b_out = attn_layer.out_proj.bias

    # Q, K, V = x_input * W_Q, W_K, W_V
    qkv = F.linear(x_input, W_in, b_in)   # shape: (B, seq_len, 3*D)
    q, k, v = torch.chunk(qkv, chunks=3, dim=-1)  # each shape (B, seq_len, D)

    # ----- 2) RESHAPE for multi‐head -----
    # (B, seq_len, D) -> (B, num_heads, seq_len, head_dim)
    q = q.view(B, seq_len, num_heads, head_dim).transpose(1, 2)  # (B, h, seq_len, head_dim)
    k = k.view(B, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, seq_len, num_heads, head_dim).transpose(1, 2)

    # ----- 3) SCALED DOT‐PRODUCT -----
    # q @ k^T, then softmax, then multiply by v
    # shape of q: (B, h, seq_len, head_dim)
    # let’s rename:
    B_h, h, S, d = q.shape  # B_h == B

    # scale q
    q = q * (1.0 / math.sqrt(d))

    # attn_weights: shape (B, h, S, S)
    #   (q @ k^T) => "bhqd,bhkd->bhqk"
    attn_weights = torch.einsum("bhqd, bhkd -> bhqk", q, k)

    # ----- 3a) Optional: Add attn_mask -----
    # PyTorch: attn_mask is shape (S, S) or (B*h, S, S) or (B, 1, S, S). 
    # You might need to expand or broadcast it to match (B, h, S, S).
    if attn_mask is not None:
        # Example: if attn_mask is (S, S), expand to (1,1,S,S)
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1,1,S,S)
        # or if attn_mask is (B,h,S,S) already, we just add it
        # We assume it's broadcastable
        attn_weights = attn_weights + attn_mask

    # ----- 3b) Optional: key_padding_mask (B, S) -----
    # Typically True => "ignore." Usually you do something like set attn_weights = -inf where mask is True.
    if key_padding_mask is not None:
        # shape is (B, S). We want to broadcast to (B, h, 1, S).
        # Indices that are True => set to -inf
        expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
        attn_weights = attn_weights.masked_fill(expanded_mask, float('-inf'))

    # softmax
    attn_weights = F.softmax(attn_weights, dim=-1)  # (B, h, S, S)

    # ----- 3c) Dropout on attn_weights -----
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)

    # Multiply by v
    # shape of attn_raw: (B, h, S, d)
    attn_raw = torch.einsum("bhqk, bhkd -> bhqd", attn_weights, v)

    # ----- 4) MERGE HEADS -----
    attn_raw = attn_raw.transpose(1,2).contiguous().view(B, seq_len, D)
    # This is the final multi-head concat, BEFORE out_proj

    # ----- 5) OUT_PROJ -----
    attn_out = F.linear(attn_raw, W_out, b_out)  # shape (B, seq_len, D)

    # ----- 6) Return what’s needed -----
    # If user wants attention weights, we might average across heads or keep separate
    # PyTorch defaults to returning shape (B, S, S) if average_attn_weights is True
    # Otherwise (B, h, S, S)
    attn_weights_return = None
    if need_weights:
        if average_attn_weights:
            # average across heads
            attn_weights_return = attn_weights.mean(dim=1)  # shape (B, S, S)
        else:
            attn_weights_return = attn_weights             # shape (B, h, S, S)

    return attn_out, attn_raw, attn_weights_return

class VisionTransformerSingle(nn.Module):
    def __init__(self,
                 ntasks,
                 img_size=224,       # 84 / 224 / 
                 patch_size=16,      # 7 / 16
                 in_channels=3,

                 emb_dim=768,       # embedding dimension of each patch (token)     128 / 512 / 768 / 192
                 depth=12,           # number of repeated transformer blocks         6 / 24   / 12
                 num_heads=12,       # number of attention heads                     4 / 16   / 12
                                    # MAKE SURE CHANGING LAYER NUMBER IN THE MAIN CODE      25 / 97

                 mlp_ratio=4.0,   # expansion ratio for the MLP
                 drop_prob=0.1
                 ):
        super().__init__()
        self.act = OrderedDict()   # This dict will store all activations needed

        # ---- Patch embedding ----
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)  # e.g. 12*12=144 / number of patches per image
        self.patch_dim = in_channels * patch_size * patch_size                 # e.g. 3*7*7=147  / flattened patch dimension

        # Linear projection for flattened patches
        self.patch_proj = nn.Linear(self.patch_dim, emb_dim)        # shape: (emb_dim, patch_dim)

        # Class token and positional embedding (1-D learnable vectors)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))         # shape: (1,1,128)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, emb_dim))
        self.pos_drop = nn.Dropout(p=drop_prob)

        # ---- Build repeated blocks for attention ----
        # We'll store all LN & linear layers in lists/ModuleLists
        self.depth = depth

        # For each block i in [0..depth-1], we define:
        #   norm1_i
        #   attn_i  (MultiheadAttention)
        #   norm2_i
        #   mlp_fc1_i
        #   mlp_fc2_i
        # We'll hold them in ModuleList for easy iteration
        self.norm1_list = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(depth)])
        self.attn_list  = nn.ModuleList([nn.MultiheadAttention(emb_dim, num_heads, dropout=drop_prob, batch_first=True) for _ in range(depth)])
        self.norm2_list = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(depth)])

        hidden_dim = int(emb_dim * mlp_ratio)       # hidden dimension of the MLP
        self.mlp_fc1_list = nn.ModuleList([nn.Linear(emb_dim, hidden_dim) for _ in range(depth)])
        self.mlp_fc2_list = nn.ModuleList([nn.Linear(hidden_dim, emb_dim) for _ in range(depth)])
        self.dropout = nn.Dropout(drop_prob)
        self.gelu = nn.GELU()

        # Final norm before the classification head
        self.norm_final = nn.LayerNorm(emb_dim)

        # ---- Task-specific heads (one per task) ----
        self.ntasks = ntasks
        self.linear = nn.ModuleList()
        for t_id, n_cls in ntasks:
            while len(self.linear) <= t_id:  # Ensure we have a linear layer for each task
                self.linear.append(None)  # Placeholder
            self.linear[t_id] = nn.Linear(emb_dim, n_cls, bias=False)       # final classification linear layer / shape: (emb_dim, n_cls) / don't peform gradient projection

        # Initialize learnable parameters
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # The other modules typically have default init that is fine

    def forward(self, x):
        """
        x shape: [B, in_channels, img_size, img_size]
        Returns a list of length len(self.ntasks), each item = logit for that task
        """
        B = x.size(0)       # mini-batch size

        # ------------------------------------------------------
        # 1) Patchify + Flatten
        #    x -> shape (B, #patches, patch_dim)
        # ------------------------------------------------------
        # Unfold into non-overlapping patches
        # shape after unfold: (B, C, #patch_h, #patch_w, patch_h, patch_w)
        # we reorder + flatten
        unfolded = x.unfold(2, self.patch_size, self.patch_size) .unfold(3, self.patch_size, self.patch_size)  # (B,C,nH,nW,p,p)
        unfolded = unfolded.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B,nH,nW,C,p,p)
        patches = unfolded.view(B, -1, self.patch_dim)              # (B, L=144, 147)
        self.act['patch_before_proj'] = patches  # shape (B, L=144, patch_dim=147)

        # ------------------------------------------------------
        # 2) Linear patch projection # (B, L, patch_dim) => [B, L, emb_dim]
        # ------------------------------------------------------
        x_patches = self.patch_proj(patches)        

        # ------------------------------------------------------
        # 3) Add cls_token + pos_embed
        #    => shape (B, 1 + L, emb_dim)
        # ------------------------------------------------------
        cls_tok = self.cls_token.expand(B, -1, -1)        # (B,1,emb_dim)
        x_cat = torch.cat((cls_tok, x_patches), dim=1)    # (B,1+L,emb_dim)
        x_cat = x_cat + self.pos_embed[:, : x_cat.size(1), :]
        x_cat = self.pos_drop(x_cat)                      # (B,1+L=145,emb_dim)

        # ------------------------------------------------------
        # 4) Repeated Transformer blocks
        #    each block i:  LN->MHA->resdual-> LN->MLP->resdual
        # ------------------------------------------------------
        x_out = x_cat                                     # (B, 1+L=145, emb_dim=128)
        for i in range(self.depth):     # i-th block
            # (a) LN => store for attn_in_proj (size= [3*D, D]) in MultiheadAttention
            x_norm1 = self.norm1_list[i](x_out)           # (B, 1+L=145, emb_dim=128) -> (B, 1+L=145, emb_dim=128)
            self.act[f'attn_in_{i}'] = x_norm1            # (B, 1+L, emb_dim)

            # (b) MultiheadAttention forward
            #    Note: with batch_first=True, shape is (B, seq_len, D)
            # attn_out, _ = self.attn_list[i](x_norm1, x_norm1, x_norm1)
        
            attn_layer = self.attn_list[i]       # get the i-th attention layer 
            attn_out, attn_raw, attn_weights = multi_head_attention_custom(
                x_norm1,
                attn_layer,
                attn_mask=None,
                key_padding_mask=None,
                need_weights=False,
                average_attn_weights=True,
                training=True,
                dropout_p=0.1,
            )
            # W_in  = attn_layer.in_proj_weight    # shape (3*D, D)
            # b_in  = attn_layer.in_proj_bias      # shape (3*D) or None
            # W_out = attn_layer.out_proj.weight   # shape (D, D)
            # b_out = attn_layer.out_proj.bias     # shape (D) or None

            # # compute Q, K, V
            # qkv = nn.functional.linear(x_norm1, W_in, b_in)  # shape (B, 1+L=145, 3*D)
            # q,k,v = torch.chunk(qkv, chunks=3, dim=-1)  # each shape (B, 1+L=145, D)

            # # reshape for multi-head attention
            # B, seq_len, D_ = q.shape
            # h = attn_layer.num_heads
            # d = D_ // h # dimension of each head
            
            # q = q.view(B, seq_len, h, d).transpose(1,2)  # (B, h, seq_len, d)
            # k = k.view(B, seq_len, h, d).transpose(1,2)
            # v = v.view(B, seq_len, h, d).transpose(1,2)

            # # scaled dot-product attention
            # #   attn_weights = softmax((q @ k^T) / sqrt(d), dim=-1)
            # #   attn_raw = (attn_weights @ v)
            # q = q * (1.0 / math.sqrt(d))  # scale
            # attn_weights = torch.einsum("bhqd,bhkd->bhqk", q, k)  # shape (B,h,seq_q,seq_k)
            # attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            # attn_raw = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)  # (B,h,seq_len,d)

            # # apply out_proj
            # attn_out = nn.functional.linear(attn_raw, W_out, b_out)  # shape (B, 1+L=145, D=emb_dim)

            # # store attn_raw_i
            # attn_raw = attn_raw.transpose(1,2).contiguous().view(B, seq_len, D_)


            self.act[f'attn_raw_{i}'] = attn_raw   # shape (B, seq_len, D) 

            # residual
            x_res1 = x_out + attn_out

            # (c) LN => store for MLP in-projection (size= [hidden_dim, D])
            x_norm2 = self.norm2_list[i](x_res1)
            
            # (d) MLP
            self.act[f'mlp_1_in_{i}'] = x_norm2  
            hidden = self.mlp_fc1_list[i](x_norm2)  # (B, 1+L, emb_dim) >> (B, 1+L, hidden_dim = emb_dim * mlp_ratio)
            hidden = self.gelu(hidden)
            hidden = self.dropout(hidden)
            self.act[f'mlp_2_in_{i}'] = hidden  
            out_mlp = self.mlp_fc2_list[i](hidden)  # (B, 1+L, hidden_dim = emb_dim * mlp_ratio) >> (B, 1+L, emb_dim)
            out_mlp = self.dropout(out_mlp)

            x_out = x_res1 + out_mlp  # final residual

        # ------------------------------------------------------
        # 5) Final LN on [CLS] token => classification
        # ------------------------------------------------------
        cls_out = self.norm_final(x_out[:, 0])  # shape (B, emb_dim)

        # ------------------------------------------------------
        # 6) Task-specific heads
        # ------------------------------------------------------
        y = []
        for t_id, _ in self.ntasks:
            y.append(self.linear[t_id](cls_out))

        return y

def vit_mini(task_details):
    return VisionTransformerSingle(task_details)

def get_Rmatrix_vit(args, model, device, data_in, keep_cls = False):
    model.eval()
    if(data_in.size(dim=1)==1):
        data_in = data_in.repeat(1, 3, 1, 1)

    with torch.no_grad():
        _  = model(data_in)

    mat_list = []
    # patch embedding
    if "patch_before_proj" in model.act:
        act_patch = model.act["patch_before_proj"].cpu().numpy()  # (B, L=144, D=147)
        B, L, D = act_patch.shape
        # Flatten to (D, B*L)
        act_patch = np.transpose(act_patch, (2, 0, 1)).reshape(D, -1)
        if args.projection == True: C_sketch = count_sketch_np(act_patch, C=D)  # shape (D, D)
        else: C_sketch = act_patch
        mat_list.append(C_sketch)  # shape (D, B*L)

    for i in range(model.depth):
        # 0) "attn_in_{i}" is the input to attn_list[i], shape (B, L+1, emb_dim)
        if f"attn_in_{i}" in model.act:
            act_attn = model.act[f"attn_in_{i}"].cpu().numpy()
            B, Lp, D = act_attn.shape
            act_atten = np.transpose(act_attn, (2, 0, 1)).reshape(D, -1)
            if args.projection == True: C_sketch = count_sketch_np(act_atten, C=D)  # shape (D, D)
            else: C_sketch = act_atten
            mat_list.append(C_sketch)  # shape (D, B*L)

        # 1) "attn_raw_{i}" is the intermediate output of attention, shape (B, L+1, emb_dim)
        if f"attn_raw_{i}" in model.act:
            raw_out = model.act[f"attn_raw_{i}"].cpu().numpy()  # (B, L+1, D=emb_dim)
            B, Lp, D = raw_out.shape
            # Flatten to (D, B*L)
            raw_out = np.transpose(raw_out, (2, 0, 1)).reshape(D, -1)
            if args.projection == True: C_sketch = count_sketch_np(raw_out, C=D)  # shape (D, D)
            else: C_sketch = raw_out
            mat_list.append(C_sketch)  # shape (D, B*L)

        # 2) "mlp_1_in_{i}" is the input to mlp_fc1_list[i], shape (B, L+1, emb_dim)
        if f"mlp_1_in_{i}" in model.act:
            act_1 = model.act[f"mlp_1_in_{i}"].cpu().numpy()  # (B, L+1, D=emb_dim)
            B, Lp, D = act_1.shape
            # Flatten to (D, B*L)
            act_1 = np.transpose(act_1, (2, 0, 1)).reshape(D, -1)
            if args.projection == True: C_sketch = count_sketch_np(act_1, C=D)  # shape (D, D)
            else: C_sketch = act_1
            mat_list.append(C_sketch)  # shape (D, B*L)

        # 3) "mlp_2_in_{i}" is the input to mlp_fc2_list[i], shape (B, L+1, H=hidden_dim)
        if f"mlp_2_in_{i}" in model.act:
            act_2 = model.act[f"mlp_2_in_{i}"].cpu().numpy()  # (B, L+1, H=hidden_dim)
            B, Lp, H = act_2.shape
            act_2 = np.transpose(act_2, (2, 0, 1)).reshape(H, -1)
            if args.projection == True: C_sketch = count_sketch_np(act_2, C=H)  # shape (H, H)
            else: C_sketch = act_2
            mat_list.append(C_sketch)  # shape (H, B*L)

    # Create list of layer names
    layer_names = ["patch_before_proj"]
    for i in range(model.depth):
        layer_names.append(f"attn_in_{i}")
        layer_names.append(f"attn_raw_{i}")   # new sub-layer
        layer_names.append(f"mlp_1_in_{i}")
        layer_names.append(f"mlp_2_in_{i}")

    # print("R_matrix:")
    # mem = 0
    # for i, (name, mat) in enumerate(zip(layer_names, mat_list)):
    #     print(f"Layer {i+1}: {name}, shape = {mat.shape}")
    #     n, k = mat.shape[0], mat.shape[1]
    #     layer_mem = (n*k*32)
    #     mem += layer_memory

    # print("representation matrix memory overhead:", mem / 8 / 1024 / 1024, "MB")
    
    return mat_list

def print_model_size_analysis(model, detailed=False, include_buffers=True):
    """
    Print model parameter sizes and memory usage
    
    Args:
        model: PyTorch model
        detailed: If True, print details for each parameter
        include_buffers: If True, include non-parameter buffers in calculation
    """
    # Collect parameter stats
    total_params = 0
    total_trainable_params = 0
    total_size_bytes = 0
    param_details = []
    
    # Get bytes per element based on dtype
    def get_bytes_per_element(dtype):
        if dtype == torch.float32 or dtype == torch.float:
            return 4
        elif dtype == torch.float64 or dtype == torch.double:
            return 8
        elif dtype == torch.float16 or dtype == torch.half:
            return 2
        elif dtype == torch.int8 or dtype == torch.uint8:
            return 1
        elif dtype == torch.int16 or dtype == torch.short:
            return 2
        elif dtype == torch.int32 or dtype == torch.int:
            return 4
        elif dtype == torch.int64 or dtype == torch.long:
            return 8
        elif dtype == torch.bool:
            return 1
        else:
            return 4  # Default to float32 size
    
    # Analyze each parameter
    for name, param in model.named_parameters():
        param_count = param.numel()
        bytes_per_elem = get_bytes_per_element(param.dtype)
        param_size_bytes = param_count * bytes_per_elem
        
        trainable = param.requires_grad
        total_params += param_count
        if trainable:
            total_trainable_params += param_count
        total_size_bytes += param_size_bytes
        
        param_details.append((name, param.shape, param_count, param_size_bytes, trainable))
    
    # Add buffers if requested (non-parameter tensors like running stats in BatchNorm)
    if include_buffers:
        for name, buffer in model.named_buffers():
            buffer_count = buffer.numel()
            bytes_per_elem = get_bytes_per_element(buffer.dtype)
            buffer_size_bytes = buffer_count * bytes_per_elem
            
            total_params += buffer_count
            total_size_bytes += buffer_size_bytes
            
            param_details.append((name, buffer.shape, buffer_count, buffer_size_bytes, False))
    
    # Calculate memory overhead
    param_memory_mb = total_size_bytes / (1024 * 1024)
    gradient_memory_mb = param_memory_mb  # Same size as parameters (for trainable parameters)
    optimizer_memory_mb = param_memory_mb * 2  # Approximate for Adam (momentum and variance)
    
    total_memory_mb = param_memory_mb + gradient_memory_mb + optimizer_memory_mb
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"MODEL MEMORY ANALYSIS")
    print(f"{'='*80}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {total_trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - total_trainable_params:,}")
    print(f"Parameter memory: {param_memory_mb:.2f} MB")
    print(f"Gradient memory (approx): {gradient_memory_mb:.2f} MB")
    print(f"Optimizer state memory (approx, Adam): {optimizer_memory_mb:.2f} MB")
    print(f"Total memory: {total_memory_mb:.2f} MB")
    print(f"{'='*80}")
    
    # Print detailed breakdown if requested
    if detailed:
        print("\nDETAILED PARAMETER BREAKDOWN:")
        print(f"{'Name':<40} {'Shape':<20} {'Parameters':>12} {'Size':>10} {'Trainable':<10}")
        print("-" * 100)
        
        for name, shape, count, size, trainable in sorted(param_details, key=lambda x: x[3], reverse=True):
            shape_str = str(tuple(shape))
            print(f"{name:<40} {shape_str:<20} {count:>12,} {size/1024:.2f} KB {'Yes' if trainable else 'No':<10}")
    
    # Group by layer types
    if detailed:
        print("\nPARAMETER GROUPS:")
        groups = {
            "attention": 0,
            "mlp": 0, 
            "normalization": 0,
            "embedding": 0,
            "classifier": 0,
            "other": 0
        }
        
        for name, _, count, _, _ in param_details:
            if "attn" in name:
                groups["attention"] += count
            elif "mlp" in name or "fc" in name:
                groups["mlp"] += count
            elif "norm" in name:
                groups["normalization"] += count
            elif "embed" in name or "token" in name or "patch" in name:
                groups["embedding"] += count
            elif "linear" in name or "head" in name:
                groups["classifier"] += count
            else:
                groups["other"] += count
        
        print(f"{'Group':<15} {'Parameters':>12} {'Percentage':>10}")
        print("-" * 40)
        for group, count in sorted(groups.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"{group:<15} {count:>12,} {count/total_params*100:>9.2f}%")

    # representation matrix analysis
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_in = torch.randn(100, 3, 84, 84).to(device)  # Example input
    model.to(device)
    args = type('', (), {})()  # Create a dummy args object
    args.projection = False
    representation_matrix = get_Rmatrix_vit(args, model,device,data_in,keep_cls = False)

    # print representation matrix
    print("\nRepresentation Matrix:")
    for i, mat in enumerate(representation_matrix):
        print(f"Layer {i+1}: shape = {mat.shape}")

    # gpm size calculation
    gpm_size = 0
    for i, mat in enumerate(representation_matrix):
        layer_size = mat.shape[0] * mat.shape[0] * 4 / (1024 * 1024)
        gpm_size += layer_size
    print(f"\nGPM size: {gpm_size:.2f} MB")

# Example usage
if __name__ == "__main__":
    # Example with your ViT model
    model = vit_mini([(task, int(100 / 10)) for task in range(10)])
    print_model_size_analysis(model)


# Convenience helper: pull weights straight from a timm model by variant name
def load_pretrained_from_timm(model: VisionTransformerSingle, variant: str, *, pretrained: bool = True, **kwargs):
    """
    Example:
        load_pretrained_from_timm(my_model, 'vit_base_patch16_224')
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError("Please `pip install timm` to use load_pretrained_from_timm.") from e
    tm = timm.create_model(variant, pretrained=pretrained, **kwargs)
    return load_pretrained_vit_into_vitsingle(model, tm.state_dict(), interpolate_pos_encoding=True, verbose=True)
# ===== End of loader =====

@torch.no_grad()
def load_pretrained_vit_into_vitsingle(
    model: VisionTransformerSingle,
    source: Union[str, Dict[str, torch.Tensor], torch.nn.Module],
    *,
    interpolate_pos_encoding: bool = True,
    strict: bool = False,
    verbose: bool = True
):
    """
    Load a timm/DeiT-style ViT checkpoint into VisionTransformerSingle.
    - 'source' can be a path to .pth, a state_dict, or a timm model.

    Returns: dict with 'copied' and 'skipped' lists for quick inspection.
    """
    sd = _to_state_dict(source)
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    copied, skipped = [], []

    # ---- patch embedding: Conv2d -> Linear (flatten kernel) ----
    pe_w_key = 'patch_embed.proj.weight'
    pe_b_key = 'patch_embed.proj.bias'
    if pe_w_key in sd:
        W = sd[pe_w_key]
        b = sd.get(pe_b_key, None)
        if W.ndim == 4:
            E, C, Ph, Pw = W.shape
            want_E, want_in = model.patch_proj.weight.shape
            if E == want_E and (C * model.patch_size * model.patch_size) == want_in and Ph == Pw == model.patch_size:
                model.patch_proj.weight.copy_(W.flatten(1).to(dtype=dtype, device=device))
                if model.patch_proj.bias is not None and b is not None:
                    model.patch_proj.bias.copy_(b.to(dtype=dtype, device=device))
                copied.append('patch_proj')
            else:
                skipped.append(f'patch_proj (shape mismatch: src {tuple(W.shape)} -> dst {tuple(model.patch_proj.weight.shape)})')
        elif W.ndim == 2 and W.shape == model.patch_proj.weight.shape:
            model.patch_proj.weight.copy_(W.to(dtype=dtype, device=device))
            if model.patch_proj.bias is not None and pe_b_key in sd:
                model.patch_proj.bias.copy_(sd[pe_b_key].to(dtype=dtype, device=device))
            copied.append('patch_proj')
        else:
            skipped.append('patch_proj (unsupported weight shape)')
    else:
        skipped.append('patch_proj (missing in checkpoint)')

    # ---- cls_token ----
    if 'cls_token' in sd and sd['cls_token'].shape == model.cls_token.shape:
        model.cls_token.copy_(sd['cls_token'].to(dtype=dtype, device=device))
        copied.append('cls_token')
    else:
        skipped.append('cls_token')

    # ---- pos_embed with interpolation if needed ----
    pe_key = _find_pos_embed_key(sd)
    if pe_key is not None:
        pos_src = sd[pe_key].to(dtype=dtype, device=device)  # (1, src_tokens, C)
        if pos_src.shape[-1] != model.pos_embed.shape[-1]:
            skipped.append(f'pos_embed (C mismatch: src {pos_src.shape[-1]} vs dst {model.pos_embed.shape[-1]})')
        else:
            if interpolate_pos_encoding:
                pos_new = _interpolate_pos_embed(
                    pos_src,
                    dst_num_patches=model.num_patches,
                    dst_extra_tokens=model.pos_embed.shape[1] - model.num_patches
                )
            else:
                pos_new = pos_src
            if pos_new.shape == model.pos_embed.shape:
                model.pos_embed.copy_(pos_new)
                copied.append('pos_embed')
            else:
                skipped.append(f'pos_embed (shape mismatch after interp: got {tuple(pos_new.shape)}, want {tuple(model.pos_embed.shape)})')
    else:
        skipped.append('pos_embed (not found)')

    # ---- transformer blocks ----
    src_depth = _num_blocks_from_state_dict(sd)
    dst_depth = model.depth
    n = min(src_depth, dst_depth)
    for i in range(n):
        # Norm1
        for name_src, mod_dst in [
            (f'blocks.{i}.norm1', model.norm1_list[i]),
            (f'blocks.{i}.norm2', model.norm2_list[i]),
        ]:
            w_key, b_key = name_src + '.weight', name_src + '.bias'
            if w_key in sd and b_key in sd and \
               sd[w_key].shape == mod_dst.weight.shape and sd[b_key].shape == mod_dst.bias.shape:
                mod_dst.weight.copy_(sd[w_key].to(dtype=dtype, device=device))
                mod_dst.bias.copy_(sd[b_key].to(dtype=dtype, device=device))
                copied.append(name_src)
            else:
                skipped.append(name_src)

        # Attention qkv / proj
        qkv_w = f'blocks.{i}.attn.qkv.weight'
        qkv_b = f'blocks.{i}.attn.qkv.bias'
        proj_w = f'blocks.{i}.attn.proj.weight'
        proj_b = f'blocks.{i}.attn.proj.bias'

        attn = model.attn_list[i]
        if qkv_w in sd and sd[qkv_w].shape == attn.in_proj_weight.shape:
            attn.in_proj_weight.copy_(sd[qkv_w].to(dtype=dtype, device=device))
            copied.append(f'blocks.{i}.attn.qkv.weight -> attn.in_proj_weight')
        else:
            skipped.append(f'blocks.{i}.attn.qkv.weight')

        if qkv_b in sd and attn.in_proj_bias is not None and sd[qkv_b].shape == attn.in_proj_bias.shape:
            attn.in_proj_bias.copy_(sd[qkv_b].to(dtype=dtype, device=device))
            copied.append(f'blocks.{i}.attn.qkv.bias -> attn.in_proj_bias')
        else:
            skipped.append(f'blocks.{i}.attn.qkv.bias')

        if proj_w in sd and sd[proj_w].shape == attn.out_proj.weight.shape:
            attn.out_proj.weight.copy_(sd[proj_w].to(dtype=dtype, device=device))
            copied.append(f'blocks.{i}.attn.proj.weight -> attn.out_proj.weight')
        else:
            skipped.append(f'blocks.{i}.attn.proj.weight')

        if proj_b in sd and attn.out_proj.bias is not None and sd[proj_b].shape == attn.out_proj.bias.shape:
            attn.out_proj.bias.copy_(sd[proj_b].to(dtype=dtype, device=device))
            copied.append(f'blocks.{i}.attn.proj.bias -> attn.out_proj.bias')
        else:
            skipped.append(f'blocks.{i}.attn.proj.bias')

        # MLP
        fc1_w, fc1_b = f'blocks.{i}.mlp.fc1.weight', f'blocks.{i}.mlp.fc1.bias'
        fc2_w, fc2_b = f'blocks.{i}.mlp.fc2.weight', f'blocks.{i}.mlp.fc2.bias'
        if fc1_w in sd and sd[fc1_w].shape == model.mlp_fc1_list[i].weight.shape:
            model.mlp_fc1_list[i].weight.copy_(sd[fc1_w].to(dtype=dtype, device=device))
            copied.append(f'blocks.{i}.mlp.fc1.weight')
        else:
            skipped.append(f'blocks.{i}.mlp.fc1.weight')
        if fc1_b in sd and sd[fc1_b].shape == model.mlp_fc1_list[i].bias.shape:
            model.mlp_fc1_list[i].bias.copy_(sd[fc1_b].to(dtype=dtype, device=device))
            copied.append(f'blocks.{i}.mlp.fc1.bias')
        else:
            skipped.append(f'blocks.{i}.mlp.fc1.bias')
        if fc2_w in sd and sd[fc2_w].shape == model.mlp_fc2_list[i].weight.shape:
            model.mlp_fc2_list[i].weight.copy_(sd[fc2_w].to(dtype=dtype, device=device))
            copied.append(f'blocks.{i}.mlp.fc2.weight')
        else:
            skipped.append(f'blocks.{i}.mlp.fc2.weight')
        if fc2_b in sd and sd[fc2_b].shape == model.mlp_fc2_list[i].bias.shape:
            model.mlp_fc2_list[i].bias.copy_(sd[fc2_b].to(dtype=dtype, device=device))
            copied.append(f'blocks.{i}.mlp.fc2.bias')
        else:
            skipped.append(f'blocks.{i}.mlp.fc2.bias')

    # ---- final norm ----
    if 'norm.weight' in sd and 'norm.bias' in sd \
       and sd['norm.weight'].shape == model.norm_final.weight.shape:
        model.norm_final.weight.copy_(sd['norm.weight'].to(dtype=dtype, device=device))
        model.norm_final.bias.copy_(sd['norm.bias'].to(dtype=dtype, device=device))
        copied.append('norm_final')
    else:
        skipped.append('norm_final')

    if verbose:
        # Filter out expected skips like classifier head
        skipped_clean = [k for k in skipped if not (k.startswith('head') or 'classifier' in k)]
        print(f"[ViT preload] Copied: {len(copied)} tensors")
        if skipped_clean:
            print(f"[ViT preload] Skipped ({len(skipped_clean)}):")
            for s in skipped_clean:
                print("  -", s)

    return {'copied': copied, 'skipped': skipped}