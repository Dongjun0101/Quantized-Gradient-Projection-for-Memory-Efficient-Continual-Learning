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
    # 1) random hash function h(i) in [0..C-1]
    h = rng.randint(low=0, high=C, size=N)  # shape (N,)
    # 2) random sign function s(i) in {+1, -1}
    signs = rng.choice([-1, +1], size=N)

    # 3) Initialize output
    out = np.zeros((D, C), dtype=mat.dtype)

    # 4) accumulate
    for i in range(N):
        bucket = h[i]
        out[:, bucket] += signs[i] * mat[:, i]

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
                 img_size=84,
                 patch_size=7,
                 in_channels=3,
                 emb_dim=128,     # embedding dimension of each patch (token)
                 depth=6,         # number of repeated transformer blocks
                 num_heads=4,     # number of attention heads
                 mlp_ratio=4.0,   # expansion ratio for the MLP
                 drop_prob=0.1):
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

def vit_cifar(task_details):
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
        else : C_sketch = act_patch

        mat_list.append(C_sketch)  # shape (D, B*L)

    for i in range(model.depth):
        # 0) "attn_in_{i}" is the input to attn_list[i], shape (B, L+1, emb_dim)
        if f"attn_in_{i}" in model.act:
            act_attn = model.act[f"attn_in_{i}"].cpu().numpy()
            B, Lp, D = act_attn.shape
            act_atten = np.transpose(act_attn, (2, 0, 1)).reshape(D, -1)
            if args.projection == True: C_sketch = count_sketch_np(act_atten, C=D)  # shape (D, D)
            else : C_sketch = act_atten
            mat_list.append(C_sketch)  # shape (D, B*L)

        # 1) "attn_raw_{i}" is the intermediate output of attention, shape (B, L+1, emb_dim)
        if f"attn_raw_{i}" in model.act:
            raw_out = model.act[f"attn_raw_{i}"].cpu().numpy()  # (B, L+1, D=emb_dim)
            B, Lp, D = raw_out.shape
            # Flatten to (D, B*L)
            raw_out = np.transpose(raw_out, (2, 0, 1)).reshape(D, -1)
            if args.projection == True: C_sketch = count_sketch_np(raw_out, C=D)  # shape (D, D)
            else : C_sketch = raw_out
            mat_list.append(C_sketch)  # shape (D, B*L)

        # 2) "mlp_1_in_{i}" is the input to mlp_fc1_list[i], shape (B, L+1, emb_dim)
        if f"mlp_1_in_{i}" in model.act:
            act_1 = model.act[f"mlp_1_in_{i}"].cpu().numpy()  # (B, L+1, D=emb_dim)
            B, Lp, D = act_1.shape
            # Flatten to (D, B*L)
            act_1 = np.transpose(act_1, (2, 0, 1)).reshape(D, -1)
            if args.projection == True: C_sketch = count_sketch_np(act_1, C=D)  # shape (D, D)
            else : C_sketch = act_1
            mat_list.append(C_sketch)  # shape (D, B*L)

        # 3) "mlp_2_in_{i}" is the input to mlp_fc2_list[i], shape (B, L+1, H=hidden_dim)
        if f"mlp_2_in_{i}" in model.act:
            act_2 = model.act[f"mlp_2_in_{i}"].cpu().numpy()  # (B, L+1, H=hidden_dim)
            B, Lp, H = act_2.shape
            act_2 = np.transpose(act_2, (2, 0, 1)).reshape(H, -1)
            if args.projection == True: C_sketch = count_sketch_np(act_2, C=H)  # shape (H, H)
            else : C_sketch = act_2
            mat_list.append(C_sketch)  # shape (H, B*L)

    # Create list of layer names
    layer_names = ["patch_before_proj"]
    for i in range(model.depth):
        layer_names.append(f"attn_in_{i}")
        layer_names.append(f"attn_raw_{i}")   # new sub-layer
        layer_names.append(f"mlp_1_in_{i}")
        layer_names.append(f"mlp_2_in_{i}")
    print("R_matrix:")
    for i, (name, mat) in enumerate(zip(layer_names, mat_list)):
        print(f"Layer {i+1}: {name}, shape = {mat.shape}")
    
    return mat_list

def print_model_size_analysis(model, detailed=True, include_buffers=True):
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

# Example usage
if __name__ == "__main__":
    # Example with your ViT model
    model = vit_cifar([(task, int(100 / 10)) for task in range(10)])
    print_model_size_analysis(model)

# R_matrices with layer names:
# Layer 1: patch_before_proj, shape = (147, 14400)

# Layer 2: attn_in_0, shape = (128, 14500)
# Layer 3: mlp_1_in_0, shape = (128, 14500)
# Layer 4: mlp_2_in_0, shape = (512, 14500)

# Layer 5: attn_in_1, shape = (128, 14500)
# Layer 6: mlp_1_in_1, shape = (128, 14500)
# Layer 7: mlp_2_in_1, shape = (512, 14500)

# Layer 8: attn_in_2, shape = (128, 14500)
# Layer 9: mlp_1_in_2, shape = (128, 14500)
# Layer 10: mlp_2_in_2, shape = (512, 14500)

# Layer 11: attn_in_3, shape = (128, 14500)
# Layer 12: mlp_1_in_3, shape = (128, 14500)
# Layer 13: mlp_2_in_3, shape = (512, 14500)

# Layer 14: attn_in_4, shape = (128, 14500)
# Layer 15: mlp_1_in_4, shape = (128, 14500)
# Layer 16: mlp_2_in_4, shape = (512, 14500)

# Layer 17: attn_in_5, shape = (128, 14500)
# Layer 18: mlp_1_in_5, shape = (128, 14500)
# Layer 19: mlp_2_in_5, shape = (512, 14500)

'''
# model weight to be projected:
pos_embed: torch.Size([1, 145, 128])

attn_list.0.in_proj_weight: torch.Size([384, 128])
attn_list.0.out_proj.weight: torch.Size([128, 128])
attn_list.1.in_proj_weight: torch.Size([384, 128])
attn_list.1.out_proj.weight: torch.Size([128, 128])
attn_list.2.in_proj_weight: torch.Size([384, 128])
attn_list.2.out_proj.weight: torch.Size([128, 128])
attn_list.3.in_proj_weight: torch.Size([384, 128])
attn_list.3.out_proj.weight: torch.Size([128, 128])
attn_list.4.in_proj_weight: torch.Size([384, 128])
attn_list.4.out_proj.weight: torch.Size([128, 128])
attn_list.5.in_proj_weight: torch.Size([384, 128])
attn_list.5.out_proj.weight: torch.Size([128, 128])

mlp_fc1_list.0.weight: torch.Size([512, 128])
mlp_fc1_list.1.weight: torch.Size([512, 128])
mlp_fc1_list.2.weight: torch.Size([512, 128])
mlp_fc1_list.3.weight: torch.Size([512, 128])
mlp_fc1_list.4.weight: torch.Size([512, 128])
mlp_fc1_list.5.weight: torch.Size([512, 128])
mlp_fc2_list.0.weight: torch.Size([128, 512])
mlp_fc2_list.1.weight: torch.Size([128, 512])
mlp_fc2_list.2.weight: torch.Size([128, 512])
mlp_fc2_list.3.weight: torch.Size([128, 512])
mlp_fc2_list.4.weight: torch.Size([128, 512])
mlp_fc2_list.5.weight: torch.Size([128, 512])
'''


'''
# Entire model weights (excluding biases):
cls_token: torch.Size([1, 1, 128])
pos_embed: torch.Size([1, 145, 128])
patch_proj.weight: torch.Size([128, 147])
norm1_list.0.weight: torch.Size([128])
norm1_list.1.weight: torch.Size([128])
norm1_list.2.weight: torch.Size([128])
norm1_list.3.weight: torch.Size([128])
norm1_list.4.weight: torch.Size([128])
norm1_list.5.weight: torch.Size([128])
attn_list.0.in_proj_weight: torch.Size([384, 128])
attn_list.0.out_proj.weight: torch.Size([128, 128])
attn_list.1.in_proj_weight: torch.Size([384, 128])
attn_list.1.out_proj.weight: torch.Size([128, 128])
attn_list.2.in_proj_weight: torch.Size([384, 128])
attn_list.2.out_proj.weight: torch.Size([128, 128])
attn_list.3.in_proj_weight: torch.Size([384, 128])
attn_list.3.out_proj.weight: torch.Size([128, 128])
attn_list.4.in_proj_weight: torch.Size([384, 128])
attn_list.4.out_proj.weight: torch.Size([128, 128])
attn_list.5.in_proj_weight: torch.Size([384, 128])
attn_list.5.out_proj.weight: torch.Size([128, 128])
norm2_list.0.weight: torch.Size([128])
norm2_list.1.weight: torch.Size([128])
norm2_list.2.weight: torch.Size([128])
norm2_list.3.weight: torch.Size([128])
norm2_list.4.weight: torch.Size([128])
norm2_list.5.weight: torch.Size([128])
mlp_fc1_list.0.weight: torch.Size([512, 128])
mlp_fc1_list.1.weight: torch.Size([512, 128])
mlp_fc1_list.2.weight: torch.Size([512, 128])
mlp_fc1_list.3.weight: torch.Size([512, 128])
mlp_fc1_list.4.weight: torch.Size([512, 128])
mlp_fc1_list.5.weight: torch.Size([512, 128])
mlp_fc2_list.0.weight: torch.Size([128, 512])
mlp_fc2_list.1.weight: torch.Size([128, 512])
mlp_fc2_list.2.weight: torch.Size([128, 512])
mlp_fc2_list.3.weight: torch.Size([128, 512])
mlp_fc2_list.4.weight: torch.Size([128, 512])
mlp_fc2_list.5.weight: torch.Size([128, 512])
norm_final.weight: torch.Size([128])
linear.0.weight: torch.Size([10, 128])
linear.1.weight: torch.Size([10, 128])
linear.2.weight: torch.Size([10, 128])
linear.3.weight: torch.Size([10, 128])
linear.4.weight: torch.Size([10, 128])
linear.5.weight: torch.Size([10, 128])
linear.6.weight: torch.Size([10, 128])
linear.7.weight: torch.Size([10, 128])
linear.8.weight: torch.Size([10, 128])
linear.9.weight: torch.Size([10, 128])
'''