import argparse
from itertools import filterfalse
import os
import time
import numpy as np
import copy
import sys
import random
import torch.backends.cudnn  # type: ignore
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
from math import ceil
from random import Random
from utils import notmnist_setup
from utils import miniimagenet_setup
from utils.utility_function import *
from scipy.linalg import svd  # Use SciPy's SVD function  # pyright: ignore[reportUnusedImport, reportMissingImports]
from scipy.stats import norm  # pyright: ignore[reportMissingImports]
from scipy.stats import kurtosis  # pyright: ignore[reportMissingImports]
from scipy.stats import shapiro  # pyright: ignore[reportMissingImports]
from PIL import Image
from torch.utils.data import Dataset
import resource
# Importing modules related to your specific implementations
from models import *
import wandb
import re
import timm
import psutil

def str2bool(v):
    return v.lower() in ("true")

# Argument parsing alexnet
parser = argparse.ArgumentParser(description='Proper AlexNet for CIFAR10/CIFAR100 in pytorch')
parser.add_argument('--run_no', default=1, type=str, help='parallel run number, models saved as model_{rank}_{run_no}.th')
parser.add_argument('--save-dir', dest='save_dir', help='The directory used to save the trained models', default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=5)
parser.add_argument('--biased', dest='biased', action='store_true', help='biased compression')
parser.add_argument('--unbiased', dest='biased', action='store_false', help='biased compression')
parser.add_argument('--level', default=32, type=int, metavar='k', help='quantization level 1-32')
parser.add_argument('--compress', default=False, type=str2bool, metavar='COMP', help='True: compress by sending coefficients associated with the orthogonal basis space')
parser.add_argument('--device', default=0, type=int, help='GPU device ID')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--port', dest='port', help='between 3000 to 65000', default='29500', type=str)  # 29500
parser.add_argument('--method', default='svd', type=str, help='gram or svd')
# In centralized setting, skew == 0, num_client == 1, frac == 1
parser.add_argument('--skew', default=0.0, type=float, help='belongs to [0,1] where 0= completely iid and 1=completely non-iid')
parser.add_argument('--n_clients', default=1, type=int, help='total number of nodes; 10')  # number of nodes. 
parser.add_argument('--frac', default=1, type=float, help='fraction of client to be updated')  # 1.0
parser.add_argument('--global_rounds', default=1, type=int)  

parser.add_argument('--momentum', default=0.0, type=float, metavar='M', help='momentum; resnet = 0.9 / alexnet = 0.0')
parser.add_argument('--deterministic', default=True, type=str2bool)  # deterministic behavior
parser.add_argument('--weight_decay', '--wd', default=0, type=float, metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--arch', '-a', metavar='ARCH', default='vit', help='alexnet / alex_quarter / resnet / vit')
parser.add_argument('--dataset', dest='dataset', default='imagenet_r', help='available datasets: miniimagenet / domainnet_hf',  type=str)  # make sure to check the number of classes below
parser.add_argument('--classes', default=100, type=int, help='number of classes in the dataset')  # miniimagenet : 100, cifar100 : 100, mnist and cifar10 : 10
parser.add_argument('--num_tasks', default=10, type=int, help='number of tasks (over time)')  # CIFAR-100 split into 10 tasks, Each task having 10 classes
parser.add_argument('--print_times', default=5, type=int)

parser.add_argument('--wandbflag', default=True, type=str2bool)  # Wandb enable
parser.add_argument('--gpmflag', default=True, type=str2bool)  # gpm enable
parser.add_argument('--projection', default=True, type=str2bool, help='projection of the representation')  # defualt = True
parser.add_argument('--seed', default=7, type=int, help='set seed, defualt = 1 4 7 10 13 16 19 22')

# In centralized setting, local_epoch = 50, global_round = 1
parser.add_argument('--threshold', default=0.75, type=float, help='threshold for the gradient memory; 0.9 -> alexnet, 0.965-> resnet')  # Similar to GPM-Codebase, default = 0.9
parser.add_argument('--increment_th', default=0.001, type=float, help='increase threshold linearly across tasks; 0.001 -> alexnet, 0 -> resnet') # default = 0.001
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 128)')
# parser.add_argument('--pretrain_lr', default=0.1, type=int)   
# parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate, alexnet -> 0.01') # default = 0.01
parser.add_argument('--pretrain_lr', default=0.005, type=int)   
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, metavar='LR', help='initial learning rate, alexnet -> 0.01') # default = 0.01
parser.add_argument('--pretrain_epochs', default=1, type=int)          
parser.add_argument('--local_epochs', default=1, type=int)                                 # 2 local epoch / 150 global round or 5 local epoch / 60 global round 

parser.add_argument('--quantization_flag', default= True   , type=str2bool)                   # false : full-precision, true : quantization
parser.add_argument('--quantization_bit', default=4, type=int)                             # false : full-precision, true : quantization
parser.add_argument('--quant_error_coeff_alpha', default = 15, type = int)                 # bigger value >> allow more flexibility     5 / 15
parser.add_argument('--quant_error_coeff_beta', default = 15, type = int)                  # bigger value >> allow more flexibility
parser.add_argument('--outlier_percent', default= 2, type=int)                             # ASYMM, ILNF4, MIXED                        1 / 2      



args = parser.parse_args()

def wandb_initialization():                         
    wandb.init(
        # project="iclr_qgpm_test",
        project="iclr_qgpm_test",
        name = f"vits{args.num_tasks} {args.dataset} / batch {args.batch_size} / lr {args.lr} / epochs {args.local_epochs}  / quantization {args.quantization_flag} / gpm {args.gpmflag} / projection {args.projection} / th {args.threshold} / bit {args.quantization_bit} / alpha {args.quant_error_coeff_alpha} / outlier {args.outlier_percent} / seed {args.seed}", 
        config=args
    )

def set_seed(seed):
    # Set the random seed for Python's random module
    random.seed(seed)
    # Set the random seed for NumPy
    np.random.seed(seed)
    # Set the random seed for PyTorch (CPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # If you are using CUDA, set the seed for all CUDA GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensuring deterministic behavior
    if args.deterministic==True:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    print(f"Random seed set to: {seed}")

def load_partial_vit_weights(
    my_model: nn.Module,
    pretrained_name: str = "vit_large_patch16_224"
    ):
    # 1. Load pretrained timm model
    timm_model = timm.create_model(pretrained_name, pretrained=True)
    pretrained_state = timm_model.state_dict()
    
    # 2. Get your current (uninitialized) model's state dict
    my_state = my_model.state_dict()
    
    # 3. Copy matching keys
    for k, v in pretrained_state.items():
        # Skip patch embedding, pos_embed, cls_token, final classification head
        if any(x in k for x in ["patch_embed", "pos_embed", "cls_token", "head"]):
            continue
        
        # If shapes match, copy!
        if k in my_state and v.shape == my_state[k].shape:
            my_state[k] = v
        else:
            pass  # shape mismatch or key not found => skip
    
    # 4. Load (allow missing keys, because we skip some)
    my_model.load_state_dict(my_state, strict=False)
    print("=> Loaded partial weights from timm model:", pretrained_name)

def match_vit_layer_name(name, depth):
    """
    Return the integer index into self.importance_scaled_feature_mat that corresponds to the subspace for this parameter.
    If no match, return None.
    """
    # 0) Patch projection
    if name.startswith('patch_proj.weight'):
        # index=0
        return 0

    # 1) Look for attention parameters
    # attn_list.<i>.in_proj_weight, attn_list.<i>.out_proj.weight, ...
    attn_in = re.match(r"attn_list\.(\d+)\.in_proj_weight", name)
    attn_out = re.match(r"attn_list\.(\d+)\.out_proj.weight", name)
    # 2) MLP: mlp_fc1_list.<i>.weight, mlp_fc2_list.<i>.weight
    mlp1 = re.match(r"mlp_fc1_list\.(\d+)\.weight", name)
    mlp2 = re.match(r"mlp_fc2_list\.(\d+)\.weight", name)

    # sub-block = 1 + i*4 + offset
    # offset=0 => attn_in
    # offset=1 => attn_raw (for out_proj)
    # offset=2 => mlp_1_in
    # offset=3 => mlp_2_in

    if attn_in:
        i = int(attn_in.group(1))
        return 1 + i*4 + 0
    if attn_out:
        i = int(attn_out.group(1))
        return 1 + i*4 + 1
    if mlp1:
        i = int(mlp1.group(1))
        return 1 + i*4 + 2
    if mlp2:
        i = int(mlp2.group(1))
        return 1 + i*4 + 3

    return None

def cosine_error_single_vector(original_vec, pseudo_quantized_vec):
    """
    1 - cosine_similarity, clamped in [0,1].
    If vectors are negatively aligned, treat that as max error = 1.
    """
    # calculate L2 norm of original and psedo quantized vector
    orig_norm = torch.norm(original_vec, p=2)
    quant_norm = torch.norm(pseudo_quantized_vec, p=2)

    # Avoid division by zero if either is all zeros
    if orig_norm == 0 or quant_norm == 0:
        return 0.0

    # Cosine similarity btw two vectors
    cos_sim = torch.sum(original_vec * pseudo_quantized_vec) / (orig_norm * quant_norm)

    # Clamp cos_sim in [0,1]
    cos_sim = torch.clamp(cos_sim, 0.0, 1.0)
    
    # print(orig_norm-quant_norm)
    if orig_norm < quant_norm:
        scaled_cos_sim = 1 - torch.clamp(args.quant_error_coeff_alpha * (1 - cos_sim), 0.0, 1.0)
    elif orig_norm >= quant_norm:
        scaled_cos_sim = 1 - torch.clamp(args.quant_error_coeff_beta * (1 - cos_sim), 0.0, 1.0)
        
    return scaled_cos_sim.item()

def columnwise_cosine_error(original_tensor, pseudo_quantized):
    """
    Return a list of cosine-based errors, one per column.
    """
    num_cols = original_tensor.size(1)
    importances = []
    for i in range(num_cols):
        col_orig = original_tensor[:, i]
        col_quant = pseudo_quantized[:, i]
        importance = cosine_error_single_vector(col_orig, col_quant)
        importances.append(importance)
    return importances

def quantize_fixed_point_ASYMM(tensor, bit_original):       # quantization function for Asymmetric quantization
    # scale                                                 
    scales = []
    zero_points = []
    r_value = tensor.size(1)
    tensor_quantized = np.zeros(tensor.shape, dtype=object)     # numpy
    pseudo_quant_tensor = torch.zeros_like(tensor)              # torch
        
    for i in range(r_value):
        bit = bit_original
        data = tensor[:, i]
        data_min = data.min()
        data_max = data.max()
        scale = (data_max - data_min) / (2**bit - 1) # scale for each column
        zero_point = torch.round(-data_min/scale).int()
        scales.append(scale.item())
        zero_points.append((int)(zero_point.item()))
        
        data_int = torch.clamp(torch.round(data/scale + zero_point).int(), 0, 2**bit - 1)
        data_int_np = data_int.cpu().numpy().astype(object)
        tensor_quantized[:, i] = data_int_np                            # numpy
        
        # psuedo quantization
        pseudo_quant_tensor[:, i] = (data_int - zero_point) * scale     # torch
        
    importances = columnwise_cosine_error(tensor, pseudo_quant_tensor)  # torch
        
    return tensor_quantized, scales, zero_points, importances

def quantize_fixed_point_ILNF4(tensor, bc):                 # quantization function for In-lier Normal float 4-bit(In-lier is quantized using NF4 and outlier is remain as float)
    # scale
    bin_centers = bc
    scales = []
    means = []
    r_value = tensor.size(1)
    tensor_quantized = np.zeros(tensor.shape, dtype=object)     # numpy
    pseudo_quant_tensor = torch.zeros_like(tensor)              # torch
    
    # Column-wise quantization, assigning each scale value to each column
    for i in range(r_value):
        data = tensor[:, i]             # Single vector
        # data_mean = data.mean()
        # centered_data = data - data_mean
        
        # q_low, q_high = torch.quantile(data, [0.01, 0.99])  # 2 percent outlier
        
        q_low = torch.quantile(data, args.outlier_percent/200)
        q_high = torch.quantile(data, 1 - args.outlier_percent/200)
        outlier_mask = (data < q_low) | (data > q_high)     # outlier_mask is True if a value in data is either below q_low or above q_high, having length of data
        outlier_values = data[outlier_mask]
        inlier_mask = ~outlier_mask
        inlier_values = data[inlier_mask]
        inlier_mean = inlier_values.mean()
        inlier_centered = inlier_values - inlier_mean
        scale = (torch.max(torch.abs(inlier_centered))).item() # scale for each column
        
        # if args.quantization_bit == 4: 
        #     q1 = torch.quantile(centered_data, 0.001)  # 0.1 percentile
        #     q3 = torch.quantile(centered_data, 0.999)  # 99.9 percentile
        #     scale = (torch.max(torch.tensor(1e-6, dtype=torch.float16), torch.max(torch.abs(q1), torch.abs(q3)))).item()
        # else:
        #     scale = (torch.max(torch.abs(centered_data))).item() # scale for each column
            
        normalized_data = torch.clamp(inlier_centered / scale, -1.0, 1.0)
        diffs = normalized_data.unsqueeze(-1) - bin_centers
        index = torch.argmin(torch.abs(diffs), dim=-1)  # keep as long by default
        
        # data_quantized = torch.zeros(tensor.size(0), dtype=torch.float32)
        # data_quantized[inlier_mask] = index.int()
        # data_quantized[outlier_mask] = outlier_values.float()
        
        scales.append(scale)
        means.append(inlier_mean.float().item())
        
        data_mixed_precision = np.zeros(data.shape, dtype=object)
        data_mixed_precision[inlier_mask.cpu().numpy()] = index.int().cpu().numpy().astype(object)  # inlier is stored in integer numpy
        data_mixed_precision[outlier_mask.cpu().numpy()] = outlier_values.float().cpu().numpy().astype(object)  # outlier is stored in float numpy
        
        tensor_quantized[:, i] = data_mixed_precision          # mixed precision numpy
        
        # psuedo quantization
        data_dequantized = torch.zeros_like(data)
        dequantized_inlier = (bin_centers[index] * scale + inlier_mean).float()         # torch
        
        data_dequantized[inlier_mask] = dequantized_inlier                              # torch
        data_dequantized[outlier_mask] = outlier_values.float()                         # torch
        
        pseudo_quant_tensor[:, i] = data_dequantized.float()                            # torch
        
    importances = columnwise_cosine_error(tensor, pseudo_quant_tensor)
        
    return tensor_quantized, scales, means, importances

def quantize_fixed_point_MIXED(tensor, bc):                     # quantization function for mixed quantization scheme (ASYMM + ILNF4)
    # scale
    bin_centers = bc
    bit = args.quantization_bit
    scales = []                 # both of the scheme store scale value in the list
    means_zps = []                  # ASYMM scheme store zero_point(ALWAYS INTEGER), ILNF4 scheme store mean value of distribution
    r_value = tensor.size(1)
    tensor_quantized = np.zeros(tensor.shape, dtype=object)     # numpy
    pseudo_quant_tensor = torch.zeros_like(tensor)              # torch
    
    # Column-wise quantization, assigning each scale value to each column
    for i in range(r_value):
        data = tensor[:, i]             # Single vector to quantize
        
        # data statistics check
        data_np = data.cpu().numpy()  # Convert tensor to numpy array
        kurt = kurtosis(data_np)
        _, p_val = shapiro(data_np)
        p_val = p_val * 10000000
        
        if (kurt < 0) and (p_val < 1) : quantizer = "ASYMM"
        else:                           quantizer = "ILNF4"
        
        # print("kurt", kurt, "p_val", p_val, "quantizer", quantizer)
        
        if quantizer == "ASYMM":        # store zero point (int) in the 'mean_or_zp' list
            data_min = data.min()
            data_max = data.max()
            scale = (data_max - data_min) / (2**bit - 1) # scale for each column
            zero_point = torch.round(-data_min/scale).int().item()
            scales.append(scale.item())
            means_zps.append((int)(zero_point))
            
            data_quantized = torch.clamp(torch.round(data/scale + zero_point).int(), 0, 2**bit - 1)
            data_int_np = data_quantized.cpu().numpy().astype(object)
            tensor_quantized[:, i] = data_int_np          # quantize each column
            
            # psuedo quantization(=dequantized matrix)
            pseudo_quant_tensor[:, i] = (data_quantized - zero_point) * scale
            
        elif quantizer == "ILNF4":     # store mean value in the 'mean_or_zp' list
            # data_mean = data.mean().float()
            # centered_data = data - data_mean
            
            # if args.quantization_bit == 4:
            #     q1 = torch.quantile(centered_data, 0.001)
            #     q3 = torch.quantile(centered_data, 0.999)
            #     scale = (torch.max(torch.tensor(1e-6, dtype=torch.float16), torch.max(torch.abs(q1), torch.abs(q3)))).item()
            # else: 
            #     scale = (torch.max(torch.abs(centered_data))).item()
            
            # q_low, q_high = torch.quantile(data, [0.01, 0.99])  # 2 percent outlier
            q_low = torch.quantile(data, args.outlier_percent/200)
            q_high = torch.quantile(data, 1 - args.outlier_percent/200)
            outlier_mask = (data < q_low) | (data > q_high)     # outlier_mask is True if a value in data is either below q_low or above q_high, having length of data
            outlier_values = data[outlier_mask]
            inlier_mask = ~outlier_mask
            inlier_values = data[inlier_mask]
            inlier_mean = inlier_values.mean()
            inlier_centered = inlier_values - inlier_mean
            scale = (torch.max(torch.abs(inlier_centered))).item()
            
            normalized_data = torch.clamp(inlier_centered / scale, -1.0, 1.0)
            diffs = normalized_data.unsqueeze(-1) - bin_centers
            index = torch.argmin(torch.abs(diffs), dim=-1)  # keep as long by default
            
            scales.append(scale)
            means_zps.append((float)(inlier_mean.item()))
            
            data_mixed_precision = np.zeros(data.shape, dtype=object)
            data_mixed_precision[inlier_mask.cpu().numpy()] = index.int().cpu().numpy().astype(object)  # inlier is stored in integer numpy
            data_mixed_precision[outlier_mask.cpu().numpy()] = outlier_values.float().cpu().numpy().astype(object)  # outlier is stored in float numpy

            tensor_quantized[:, i] = data_mixed_precision          # quantize each column
            
            
            # psuedo quantization
            data_dequantized = torch.zeros_like(data)
            dequantized_inlier = (bin_centers[index] * scale + inlier_mean).float()         # torch
            
            data_dequantized[inlier_mask] = dequantized_inlier                              # torch
            data_dequantized[outlier_mask] = outlier_values.float()                         # torch
            
            pseudo_quant_tensor[:, i] = data_dequantized.float()    
            
            
    importances = columnwise_cosine_error(tensor, pseudo_quant_tensor)
        
    return tensor_quantized, scales, means_zps, importances

# copied from codec samely
def update_GPM (self, rep_matrix, threshold):
    epsilon = 1e-10

    if self.rank == 0 : print ('Threshold of GPM: ', threshold) 
    
    # convert representation_matrix to torch tensor
    representation_matrix = []
    for i in range(len(rep_matrix)):
        # representation_matrix.append(torch.Tensor(rep_matrix[i]).to(self.device))
        representation_matrix.append(torch.Tensor(rep_matrix[i]))

    number_of_added_components = []
    if self.QGPM == []: # First task
        for i in range(len(representation_matrix)): # i = 0, 1, 2, 3, 4
            # activation = torch.Tensor(representation_matrix[i]).to(self.device)
            activation = representation_matrix[i]
            try: 
                U, S, Vh = torch.linalg.svd(activation, full_matrices=False)
                U = U.to(self.device)
            except Exception as e:
                try:
                    U, S, Vh = torch.linalg.svd(activation + epsilon, full_matrices=False)
                    U = U.to(self.device)
                except Exception as e2:
                    raise e2
            
            # criteria
            sval_total = (S.pow(2)).sum()
            sval_ratio = (S.pow(2))/sval_total
            cumulative = torch.cumsum(sval_ratio.cpu(), dim=0).to(sval_ratio.device)
            r = int((cumulative < threshold[i]).sum().item())
                        
            if args.quantization_flag == True:
                # U_int_r : object type np array
                if i == 0:                      # For the first layer : use asymmetric quantization scheme
                    U_int_r, scales_r, zp_r, importances_r = quantize_fixed_point_ASYMM(U[:,0:r], args.quantization_bit) # mean_or_zp_r is zero_point (float)
                    self.QGPM.append(U_int_r)
                    self.GPM_scales[i].extend(scales_r)             # float 
                    self.mean_or_zp_list[i].extend(zp_r)            # integer 
                    self.importance_list[i].extend(importances_r)
                    
                elif (i == 1) or (i == 2):       # For the subsequet layers(layer 1, 2) : r = 0,1,2,3 >> use asymmetric quantization scheme, subsequent r >> use ILNF4 quantization
                    U_int_r, scales_r, mean_or_zp_r, importances_r = quantize_fixed_point_MIXED(U[:,0:r], self.bin_centers)     # mean_or_zp_r is mean or zerp point (int or float)
                    self.QGPM.append(U_int_r)
                    self.GPM_scales[i].extend(scales_r)             # float
                    self.mean_or_zp_list[i].extend(mean_or_zp_r)    # mixed precision
                    self.importance_list[i].extend(importances_r)
                    
                elif i >= 3:                    # For the subsequent layers(layer 3, 4) : use ILNF4 quantization only
                    U_int_r, scales_r, mean_r, importances_r = quantize_fixed_point_ILNF4(U[:,0:r], self.bin_centers)     # mean_or_zp_r is mean value(float)
                    self.QGPM.append(U_int_r)
                    self.GPM_scales[i].extend(scales_r)             # float
                    self.mean_or_zp_list[i].extend(mean_r)          # float
                    self.importance_list[i].extend(importances_r)
                    
            elif args.quantization_flag == False:
                self.QGPM.append(U[:,0:r])
                    
            number_of_added_components.append(r)

    else:           # subsequent task
        for i in range(len(representation_matrix)):
            # activation = torch.Tensor(representation_matrix[i]).to(self.device)
            activation = representation_matrix[i]
            try: 
                U1, S1, Vh1 = torch.linalg.svd(activation, full_matrices=False) # ERROR! it says Activation contain nan or inf value
                U1 = U1.to(self.device)
            except Exception as e:
                try:
                    U1, S1, Vh1 = torch.linalg.svd(activation + epsilon, full_matrices=False)
                    U1 = U1.to(self.device)
                except Exception as e2:
                    raise e2
                    
            sval_total = (S1.pow(2)).sum()
            
            # Projected Representation 
            act_hat_torch = activation - torch.mm(self.feature_mat[i].to("cpu"), activation) # orthogonal components with respect to existing GPM            act_hat = act_hat.astype(np.float64)
            try : 
                U,S,Vh = torch.linalg.svd(act_hat_torch, full_matrices=False)
                U = U.to(self.device)
            except Exception as e:
                try:
                    U, S, Vh = torch.linalg.svd(act_hat_torch + epsilon, full_matrices=False)
                    U = U.to(self.device)
                except Exception as e2:
                    raise e2

            # criteria
            sval_hat = (S.pow(2)).sum()
            sval_ratio = (S.pow(2))/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
            
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
                
            if r == 0:
                if self.rank == 0 : print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                number_of_added_components.append(r)
                continue
            
            if args.quantization_flag == True:
                if i == 0:
                    U_int_r, scales_r, zp_r, importances_r = quantize_fixed_point_ASYMM(U[:,0:r], args.quantization_bit)
                    newGPM = np.hstack([self.QGPM[i], U_int_r])                           # Concatenate the new feature with the existing GPM
                    self.GPM_scales[i].extend(scales_r)
                    self.mean_or_zp_list[i].extend(zp_r)        # only integer
                    self.importance_list[i].extend(importances_r)
                    
                    if newGPM.shape[1] > newGPM.shape[0] : 
                        self.QGPM[i]=newGPM[:,0:newGPM.shape[0]]
                        self.GPM_scales[i] = self.GPM_scales[0: newGPM.shape[0]]
                        self.mean_or_zp_list[i] = self.mean_or_zp_list[0: newGPM.shape[0]]
                        self.importance_list[i] = self.importance_list[0: newGPM.shape[0]]
                    else                                 : 
                        self.QGPM[i]=newGPM
                    
                elif (i == 1) or (i == 2):
                    U_int_r, scales_r, mean_or_zp_r, importances_r = quantize_fixed_point_MIXED(U[:,0:r], self.bin_centers)
                    # U_int_r, scales_r, mean_or_zp_r, importances_r = quantize_fixed_point_ILNF4(U[:,0:r], self.bin_centers)
                    newGPM = np.hstack([self.QGPM[i], U_int_r])                           # Concatenate the new feature with the existing GPM
                    self.GPM_scales[i].extend(scales_r)
                    self.mean_or_zp_list[i].extend(mean_or_zp_r)# mixed precision
                    self.importance_list[i].extend(importances_r)
                    
                    if newGPM.shape[1] > newGPM.shape[0] : 
                        self.QGPM[i]=newGPM[:,0:newGPM.shape[0]]
                        self.GPM_scales[i] = self.GPM_scales[0: newGPM.shape[0]]
                        self.mean_or_zp_list[i] = self.mean_or_zp_list[0: newGPM.shape[0]]
                        self.importance_list[i] = self.importance_list[0: newGPM.shape[0]]
                    else                                 : 
                        self.QGPM[i]=newGPM
                    
                elif i >= 3:
                    U_int_r, scales_r, mean_r, importances_r = quantize_fixed_point_ILNF4(U[:,0:r], self.bin_centers)
                    newGPM = np.hstack([self.QGPM[i], U_int_r])                           # Concatenate the new feature with the existing GPM
                    self.GPM_scales[i].extend(scales_r)
                    self.mean_or_zp_list[i].extend(mean_r)      # only float
                    self.importance_list[i].extend(importances_r)
                    
                    if newGPM.shape[1] > newGPM.shape[0] : 
                        self.QGPM[i]=newGPM[:,0:newGPM.shape[0]]
                        self.GPM_scales[i] = self.GPM_scales[0: newGPM.shape[0]]
                        self.mean_or_zp_list[i] = self.mean_or_zp_list[0: newGPM.shape[0]]
                        self.importance_list[i] = self.importance_list[0: newGPM.shape[0]]
                    else                                 : 
                        self.QGPM[i]=newGPM
                    
            elif args.quantization_flag == False:   # Full precision : No quantization(GPM_scale 1), No QEA scaling(importance 1)
                newGPM = torch.cat((self.QGPM[i],U[:,0:r]), dim=1)                          # Concatenate the new feature with the existing GPM
                if newGPM.shape[1] > newGPM.shape[0] : 
                    self.QGPM[i]=newGPM[:,0:newGPM.shape[0]]
                    self.GPM_scales[i] = self.GPM_scales[0: newGPM.shape[0]]
                    self.mean_or_zp_list[i] = self.mean_or_zp_list[0: newGPM.shape[0]]
                    self.importance_list[i] = self.importance_list[0: newGPM.shape[0]]
                else                                 : 
                    self.QGPM[i]=newGPM   
                             
            number_of_added_components.append(r)

    
    if self.rank == 0:
        N = []
        K = []
        print ('Number of Added Components: ', number_of_added_components)
        print('-' * 40)
        
        print('Gradient Constraints Summary')
        print('-' * 40)
        for i in range(len(self.QGPM)):
            print ('Layer {} : {}/{}'.format(i+1, self.QGPM[i].shape[1], self.QGPM[i].shape[0]))
            N.append(self.QGPM[i].shape[0])
            K.append(self.QGPM[i].shape[1])
        print('-' * 40)

        print('Importance list Summary')
        print('-' * 40)
        for i in range(len(self.importance_list)):
            formatted_importances = [format_val(val) for val in self.importance_list[i]]
            # print('Layer {} : {}'.format(i + 1, formatted_importances))
            print("mean value of importance: ", np.mean(self.importance_list[i]))
        print('-' * 40)
        mem = 0
        p = args.outlier_percent
        for n, k in zip(N, K):  
            if args.quantization_flag == True:
                if args.quantization_bit == 8: layer_memory = (3 * k * 32) + ((n*k*(100-p)/(100)*8)+(n*k*p/100*32))
                elif args.quantization_bit == 4: layer_memory = (3 * k * 32) + ((n*k*(100-p)/(100)*4)+(n*k*p/100*32))
            else: layer_memory = (n*k*32)
            mem += layer_memory
        if args.quantization_flag == True: print("QGPM memory overhead:", mem / 8 / 1024 / 1024, "MB")
        else: print("GPM_FP memory overhead:", mem / 8 / 1024 / 1024, "MB")
        print('-' * 40)

# The Node class encapsulating the node's functionality
class Client:
    # Constructor
    def __init__(self, rank, size, device, args, train_loaders_list, val_loader): 
        self.rank = rank    # node's rank
        self.epochs = 100
        self.task_id= 0
        self.lr= 0.1
        self.size = size    # total number of nodes; i.e. network size
        self.device = device    # device(GPU) ID
        self.args = args        # passed arguments
        self.train_loaders_list = train_loaders_list  # n-th node train_loader[k-th task]
        self.val_loader = val_loader                  # val_loader[k-th task] ; shared across the all nodes
        if args.dataset == '5datasets':
            self.task_details = [(task, 10) for task in range(args.num_tasks)] # (# of task, # of classes per task)
            self.cpt = 10  # Classes per task
        else:
            self.task_details = [(task, int(args.classes / args.num_tasks)) for task in range(args.num_tasks)] # [(0, 10), (1, 10), ..., (9, 10)] )
            self.cpt = int(args.classes / args.num_tasks) # classes per task
        self.model = self.init_model()                # current node's model
        self.optimizer = self.init_optimizer()        # current node's optimizer
        self.scheduler = self.init_scheduler()        # current node's scheduler
        self.QGPM = []                                 # current node's feature list ; i.e. GPM
        self.feature_mat = []                         # M @ M^T
        self.importance_scaled_feature_mat = []                  # M @ D @ importance-D @ M^T
        if args.arch == "alexnet":
            self.no_layers = 5                            # Number of layers to consider for GPM
        elif args.arch == "resnet":
            self.no_layers = 20
        elif args.arch == "vit":
            # self.no_layers = 25
            # self.no_layers = 49
            self.no_layers = 49
        self.criterion = nn.CrossEntropyLoss()
        self.best_prec1 = 0
        self.bsz_train = None                         # train batch size; Will be set when data is partitioned
        self.bsz_val = None                           # validation batch size; Will be set when val_loader is created
        self.acc_matrix = np.zeros((args.num_tasks, args.num_tasks)) # current node's accuracy matrix
        self.GPM_scales = [[] for _ in range(self.no_layers)]
        self.importance_list = [[] for _ in range(self.no_layers)]     # hightest bit = 1
        self.mean_or_zp_list = [[] for _ in range(self.no_layers)]
        self.bin_centers = self.init_bin_centers()           # bin centers for quantization
        self.num_projection = 0
        
    def init_bin_centers(self):
        bin_centers_standard_normal = np.array([norm.ppf((i + 0.5) / (2**args.quantization_bit)) for i in range(2**args.quantization_bit)])
        min_val, max_val = bin_centers_standard_normal[0], bin_centers_standard_normal[-1]
        center_scale = 1.0 / max(abs(min_val), abs(max_val))
        bin_centers = bin_centers_standard_normal * center_scale
        bin_centers = torch.Tensor(bin_centers).to(self.device) # [-1.0  -0.70756877  -0.5422091   -0.41681885  -0.31090474  -0.21594631  0.12734098  -0.04209538   0.04209538   0.12734098   0.21594631   0.31090474  0.41681885   0.5422091   0.70756877  1.0]
        # bin_centers = torch.Tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]).to(self.device)
        if self.rank == 0: print("bin_centers: ", bin_centers)
        return bin_centers
    
    def init_model(self): # used at constructor
        # torch.manual_seed(self.args.seed + self.rank)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        if self.args.arch == 'alexnet':
            model = alexnet(self.task_details)

        elif self.args.arch == 'alex_quarter':
            model = alexnet_scaled(self.task_details)
            
        elif self.args.arch == 'resnet':
            # You can pass your dataset name, the “task_details”, and choose nf=32 or 64, etc.
            model = ResNet18_mini(self.args.dataset, self.task_details, nf=20)
            # model = ResNet18(self.task_details, nf=20)
        elif self.args.arch == 'vit':
            model = vit_mini(self.task_details)
            load_report = load_pretrained_from_timm(model, 'vit_base_patch16_224', pretrained=True)  # vit_small_patch16_224_in21k / vit_base_patch16_224 / vit_tiny_patch16_224
        else:
            raise ValueError("Unknown architecture")
        model.to(self.device)
        return model

    def init_optimizer(self): # used at init_task
        optimizer = optim.SGD(self.model.parameters(), self.lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum, nesterov=False)
        # adam optimizer
        # optimizer = optim.Adam(
        # self.model.parameters(),
        # lr=0.0001,
        # betas=(0.9, 0.999),
        # eps=1e-8,
        # weight_decay=0.00001
        # )
        return optimizer

    def init_scheduler(self): # used at init_task
        gamma = 0.1
        step1 = int(self.epochs / 2)
        step2 = int(3 / 4 * self.epochs)
        scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=gamma, milestones=[step1, step2])

        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        # self.optimizer,
        # T_max=self.args.local_epochs
        # )
        return scheduler

    def init_task(self, task_id): # called for every task
        self.task_id = task_id
        if task_id == 0:
            self.lr = self.args.pretrain_lr
            self.epochs = self.args.pretrain_epochs
        else:
            self.lr = self.args.lr                  # by default, 0.01
            self.epochs = self.args.local_epochs    # by default, 50
        
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()
        if args.gpmflag == True: self.calculate_feature_mat()

    def calculate_feature_mat(self):
        self.feature_mat = []               # used at gpm construction
        self.importance_scaled_feature_mat = []  
        int_masks = []
        float_masks = []
        int_gpms = []       # quantized gpm; to be dequantized
        float_gpms = []     # already full precision; outlier
        
        if self.args.quantization_flag == True:
            if (self.args.gpmflag == True) and (self.task_id > 0):
                for n in range(self.no_layers):
                    QGPM = copy.deepcopy(self.QGPM[n])  # copy QGPM to QGPM
                    int_mask = np.frompyfunc(lambda x: isinstance(x, int), 1, 1)(QGPM).astype(bool)
                    int_masks.append(torch.Tensor(int_mask).to(self.device))
                    float_mask = ~int_mask
                    float_masks.append(torch.Tensor(float_mask).to(self.device))
                    
                    int_gpm = np.where(int_mask, QGPM, 0).astype(float)
                    int_gpms.append(torch.Tensor(int_gpm).to(self.device))
                    float_gpm = np.where(float_mask, QGPM, 0).astype(float)
                    float_gpms.append(torch.Tensor(float_gpm).to(self.device))
                
                
                for i in range(self.no_layers):  # i = 0, 1, 2, 3, 4
                    if i == 0:       # ASYMMetric quantization >> no outlier aware quantization. every entries are integer
                        # D = torch.Tensor(np.diag(np.array(self.GPM_scales[i])**2)).to(self.device)                      # construct diagonal matrix with scales for dequantization
                        imporatnce_D = torch.Tensor(np.diag(np.array(self.importance_list[i]))).to(self.device)
                        # need to substract the zero point value from each columns of QGPM
                        scale_mat = torch.Tensor(np.array(self.GPM_scales[i])).to(self.device).unsqueeze(0)
                        zero_points = torch.Tensor(np.array(self.mean_or_zp_list[i])).to(self.device).unsqueeze(0)
                        restored_gpm = (int_gpms[i] - zero_points) * scale_mat * int_masks[i]  + float_gpms[i]
                        
                        ith_feature_mat = restored_gpm @ restored_gpm.T     # M @ D @ M^T yeilds the original feature matrix
                        self.feature_mat.append(ith_feature_mat)
                        
                        ith_importance_scaled_feature_mat = restored_gpm @ imporatnce_D @ restored_gpm.T
                        self.importance_scaled_feature_mat.append(ith_importance_scaled_feature_mat)
                        
                    elif (i == 1) or (i == 2):    # MIXED quantization
                        mean_or_zp_array = np.array(self.mean_or_zp_list[i], dtype = object)
                        
                        # boolean mask
                        # True if integer => ASYMM(zero point); False => ILNF4(mean)
                        is_int_mask = np.frompyfunc(lambda x: isinstance(x, int), 1, 1)(mean_or_zp_array).astype(bool)
                        
                        # indexs for ASYMMetric quantization
                        asymm_indexs = np.where(is_int_mask)[0]     # integer indexs
                        # indexs for ILNF4 quantization
                        ilnf4_indexs = np.where(~is_int_mask)[0]    # float indexs
                        
                        ith_feature_mat_asymm = None
                        ith_feature_mat_ilnf4 = None
                        ith_importance_scaled_feature_mat_asymm = None
                        ith_importance_scaled_feature_mat_ilnf4 = None
                        
                        # ==================== 1) ASYMM Columns ====================
                        if len(asymm_indexs) > 0:
                            QGPM_sub_asymm = int_gpms[i][:, asymm_indexs]    # extract integer indexs for ASYMM
                        
                            scales_asymm = torch.tensor(
                                [self.GPM_scales[i][c] for c in asymm_indexs],
                                device=self.device, dtype=torch.float32
                            )
                            importances_asymm = torch.tensor(
                                [self.importance_list[i][c] for c in asymm_indexs],
                                device=self.device, dtype=torch.float32
                            )
                            zero_points_asymm = torch.tensor(
                                [self.mean_or_zp_list[i][c] for c in asymm_indexs],
                                device=self.device, dtype=torch.long
                            )
                        
                            ImpD_asymm = torch.diag(importances_asymm)
                            scaleD_asymm = torch.diag(scales_asymm**2)
                            
                            # Z_asymm  = QGPM_sub_asymm - zero_points_asymm.unsqueeze(0)
                            
                            restored_gpm_asymm = (QGPM_sub_asymm - zero_points_asymm.unsqueeze(0)) * int_masks[i][:, asymm_indexs] * scales_asymm.unsqueeze(0) + float_gpms[i][:, asymm_indexs]
                            
                            ith_feature_mat_asymm = restored_gpm_asymm  @ restored_gpm_asymm.T
                            ith_importance_scaled_feature_mat_asymm = restored_gpm_asymm  @ ImpD_asymm @ restored_gpm_asymm.T
                        
                        # ==================== 2) ILNF4 Columns ====================
                        if len(ilnf4_indexs) > 0:
                            QGPM_sub_ilnf4 = int_gpms[i][:, ilnf4_indexs].long()
                            
                            scales_ilnf4 = torch.tensor(
                                [self.GPM_scales[i][c] for c in ilnf4_indexs],
                                device=self.device, dtype=torch.float32
                            )
                            importances_ilnf4 = torch.tensor(
                                [self.importance_list[i][c] for c in ilnf4_indexs],
                                device=self.device, dtype=torch.float32
                            )
                            means_ilnf4 = torch.tensor(
                                [self.mean_or_zp_list[i][c] for c in ilnf4_indexs],
                                device=self.device, dtype=torch.float32
                            )

                            ImpD_ilnf4 = torch.diag(importances_ilnf4)
                            
                            # # Convert indices to bin_centers => (n x #ilnf4)
                            # Z_ilnf4 = self.bin_centers[QGPM_sub_ilnf4]
                            # # Multiply by scale
                            # Z_ilnf4 = Z_ilnf4 * scales_ilnf4.unsqueeze(0)
                            # # Add per-column mean
                            # Z_ilnf4 = Z_ilnf4 + means_ilnf4.unsqueeze(0)
                            restored_gpm_ilnf4 = (self.bin_centers[QGPM_sub_ilnf4] * scales_ilnf4.unsqueeze(0) + means_ilnf4.unsqueeze(0)) * int_masks[i][:, ilnf4_indexs] + float_gpms[i][:, ilnf4_indexs]

                            ith_feature_mat_ilnf4 = restored_gpm_ilnf4 @ restored_gpm_ilnf4.T
                            ith_importance_scaled_feature_mat_ilnf4 = restored_gpm_ilnf4 @ ImpD_ilnf4 @ restored_gpm_ilnf4.T

                        # ==================== 3) Combine partial results ====================
                        if ith_feature_mat_asymm is not None and ith_feature_mat_ilnf4 is not None:         # Both Asymm and ILNF4 are used
                            ith_feature_mat = ith_feature_mat_asymm + ith_feature_mat_ilnf4
                            ith_importance_scaled_feature_mat = (
                                ith_importance_scaled_feature_mat_asymm + ith_importance_scaled_feature_mat_ilnf4
                            )
                        elif ith_feature_mat_asymm is not None:                                             # only Asymm is used
                            ith_feature_mat = ith_feature_mat_asymm
                            ith_importance_scaled_feature_mat = ith_importance_scaled_feature_mat_asymm
                        else:
                            # Only ILNF4 columns                                                            # only ILNF4 is used
                            ith_feature_mat = ith_feature_mat_ilnf4
                            ith_importance_scaled_feature_mat = ith_importance_scaled_feature_mat_ilnf4

                        # Finally store them
                        self.feature_mat.append(ith_feature_mat)
                        self.importance_scaled_feature_mat.append(ith_importance_scaled_feature_mat)


                    elif i >= 3:    # ILNF4 quantization
                        # self.GPM_scales[i] : 1 x m
                        # convert self.GPM_scales[i] to n x m
                        imporatnce_D = torch.Tensor(np.diag(np.array(self.importance_list[i]))).to(self.device)
                        mean_matrix = torch.Tensor(np.array(self.mean_or_zp_list[i])).to(self.device).unsqueeze(0)
                        scale_mat = torch.Tensor(np.array(self.GPM_scales[i])).to(self.device).unsqueeze(0)
                        dequantized_GPM = (self.bin_centers[int_gpms[i].long()] * scale_mat + mean_matrix) * int_masks[i] + float_gpms[i]
                        
                        ith_feature_mat = dequantized_GPM @ dequantized_GPM.T     # M @ D @M^T yeilds the original feature matrix
                        self.feature_mat.append(ith_feature_mat)
                        
                        ith_importance_scaled_feature_mat = dequantized_GPM @ imporatnce_D @ dequantized_GPM.T
                        self.importance_scaled_feature_mat.append(ith_importance_scaled_feature_mat)
                
        elif self.args.quantization_flag == False:
            GPM = [torch.Tensor(copy.deepcopy(self.QGPM[i])).to(self.device) for i in range(len(self.QGPM))]
            # Update feature_mat based on GPM
            self.feature_mat = [] # M @ M^T
            self.importance_scaled_feature_mat = []   

            if (self.args.gpmflag == True) and (self.task_id > 0):
                for i in range(len(GPM)):
                    ith_feature_mat = GPM[i] @ GPM[i].T
                    self.feature_mat.append(ith_feature_mat)
                    self.importance_scaled_feature_mat.append(ith_feature_mat)
        
        if self.rank == 0 : print('-' * 40)
                           
    def train_epoch(self): # For every epoch
        losses = AverageMeter()
        top1 = AverageMeter()
        self.model.train() # switch to train mode
        # step = len(self.train_loaders_list[self.task_id]) * self.bsz_train * epoch # num of mini batch * batch size(num of data per MB) * epoch = total num of data processed until now
        
        for batch_idx, (input, target) in enumerate(self.train_loaders_list[self.task_id]): # For every mini-batch
            input_var, target_var = input.to(self.device), (target % self.cpt).to(self.device) # moves the input batch input to the specified device (e.g., GPU), storing it in input_var
            if input_var.size(1) == 1: # if input is grayscale, expand it to 3 channels
                input_var = input_var.repeat(1,3,1,1)
            
            outputs = self.model(input_var)
            output = outputs[self.task_id]
            loss = self.criterion(output, target_var) 
            self.optimizer.zero_grad() # zero the gradient buffers
            loss.backward()            # calculate the gradient of the loss w.r.t. the model parameters
            
            # Apply GPM constraints if needed
            if (self.task_id > 0) and (self.args.gpmflag == True):
                kk = 0
                if self.args.arch == 'alexnet':
                    for k, (m, params) in enumerate(self.model.named_parameters()):
                        if k < 15 and len(params.size()) != 1:
                            sz = params.grad.data.size(0)
                            params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1), self.importance_scaled_feature_mat[kk]).view(params.size())
                            kk += 1
                        elif (k < 15 and len(params.size()) == 1) and self.task_id != 0:
                            params.grad.data.fill_(0)

                elif self.args.arch == 'resnet':
                    for k, (m,params) in enumerate(self.model.named_parameters()):
                        if len(params.size())==4:
                            sz =  params.grad.data.size(0)
                            params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1), self.importance_scaled_feature_mat[kk]).view(params.size())
                            kk+=1
                            self.num_projection += 1
                        elif len(params.size())==1 and task_id !=0:
                            params.grad.data.fill_(0)

                elif self.args.arch == 'vit':
                    for name, params in self.model.named_parameters():

                        # if no grad, skip
                        if params.grad is None:
                            continue
                        # zero out all 1-D params (bias / LN) after first task
                        if params.ndim == 1 and self.task_id != 0:
                            params.grad.data.zero_()
                            continue

                        # if param is 2D, chech if it is one of MLP weights matrices
                        if params.ndim == 2:
                            # patch projection and attention matrix (Q,K,V) also has a 2d shape: (emb_dim, patch_dim), (emb_dim, emb_dim)
                            idx = match_vit_layer_name(name, self.model.depth)
                            if idx is not None:
                                outdim = params.grad.data.size(0)  # output size
                                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(outdim, -1), self.importance_scaled_feature_mat[idx]).view(params.size())
                            
                            # if "patch_proj" in name or "attn_list" in name:
                            #     continue

                            # # mlp
                            # if "mlp_fc1_list" in name or "mlp_fc2_list" in name:
                            #     if "mlp_fc1_list" in name:
                            #         sz = params.grad.data.size(0)               # output size
                            #         params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1), self.importance_scaled_feature_mat[2*kk]).view(params.size())
                            #         kk+=1
                            #         if kk == 6: kk = 0
                            #     elif "mlp_fc2_list" in name:
                            #         sz = params.grad.data.size(0)               # output size
                            #         params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1), self.importance_scaled_feature_mat[2*kk+1]).view(params.size())
                            #         kk+=1


            self.optimizer.step() # take gradient step
            
            output = output.float()
            loss = loss.float()
            
            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            # step += self.bsz_train
            
        return self.model.state_dict(), losses.avg, top1.avg # return the model parameter after update / average loss

    def validate_seen_tasks(self):
        prec = []
        total_data_num = 0
        for tn in range(self.task_id + 1): # upto current task. assume we are on task 2, then tn = 0,1,2
            acc, loss = self.validate_task(tn)
            total_data_num += len(self.val_loader[tn])
            prec.append(acc * len(self.val_loader[tn]))
            
            if tn == self.task_id:
                current_loss = loss
                current_acc = acc
        acc_prev_tasks = sum(prec) / total_data_num
        return acc_prev_tasks, current_loss, current_acc

    def validate_task(self, task_id):
        val_loader = self.val_loader[task_id]
        top1 = AverageMeter()
        losses = AverageMeter()
        self.model.eval()

        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                input_var, target_var = input.to(self.device), (target % self.cpt).to(self.device)
                
                if input_var.size(1) == 1:
                    input_var = input_var.repeat(1, 3, 1, 1)
                
                outputs = self.model(input_var)
                output = outputs[task_id]
                loss = self.criterion(output, target_var)
                output = output.float()
                loss = loss.float()
                prec1 = accuracy(output.data, target_var)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
        return top1.avg, losses.avg

    def update_acc_matrix(self):
        for tn in range(self.task_id + 1):
            acc, _ = self.validate_task(tn)
            self.acc_matrix[self.task_id, tn] = acc
            
        return self.acc_matrix

    def print_acc_matrix(self):
        print('Node {} Overall Accuracies:'.format(self.rank))
        for i_a in range(self.task_id + 1):
            print('\t', end='')
            for j_a in range(self.acc_matrix.shape[1]):
                print('{:5.1f}% '.format(self.acc_matrix[i_a, j_a]), end='')
            print()

    def print_performance(self):
        _, _, prec1 = self.validate_seen_tasks()
        print('Node {} Task {} Accuracy: {:5.2f}%'.format(self.rank, self.task_id, prec1))

    def myGPM_update(self, threshold):
        count, data_in = 0, None
        train_loader = self.train_loaders_list[self.task_id] # using own local data
        
        # Collect training sub-sample data for GPM update
        for i, (input, target) in enumerate(train_loader):
            inp, target_in = Variable(input).to(self.device), Variable(target).to(self.device)
            
            if inp.size(1) == 1:  # If grayscale, repeat to make 3 channels
                inp = inp.repeat(1, 3, 1, 1)
            
            data_in = torch.cat((data_in, inp), 0) if data_in is not None else inp
            count += target_in.size(0)
            if count >= 100: break      # count become 128
            
        # compute local representation matrix(activation)
        
        # representation_matrix = get_representation_matrix(self.model, self.device, data_in, 4, self.rank) # 4 : layer count. defined at alexnet_model.py
        if self.args.arch == 'resnet':
            # The last Boolean parameter is for 'nodes' in your snippet; set False if not distributed
            representation_matrix = get_Rmatrix_resnet18_mini(
                args,
                self.model,
                device=self.device,
                data=data_in,
                nodes=False,
                rank=self.rank,
                dataset=self.args.dataset
            )
        elif self.args.arch == 'alexnet':
            representation_matrix = get_representation_matrix(
                self.model,
                self.device,
                data_in,
                4,           # the layer count for alexnet
                self.rank
            )
        elif self.args.arch == 'vit':
            representation_matrix = get_Rmatrix_vit(
                self.args,
                self.model,
                self.device,
                data_in,
                keep_cls = False,
            )
        
        for i in range(len(representation_matrix)):
            activation = representation_matrix[i]
            has_nan = np.isnan(activation).any()
            has_inf = np.isinf(activation).any()
            if has_nan or has_inf: 
                if has_nan: 
                    nan_count = np.isnan(activation).sum()
                    nan_indices = np.argwhere(np.isnan(activation))
                elif has_inf: 
                    inf_count = np.isinf(activation).sum()
                    inf_indices = np.argwhere(np.isinf(activation))
                raise ValueError('Node {} Task {} Layer {} - Activation matrix has nan or inf value, {} of nan or inf value starting at {}'.format(self.rank, self.task_id, i + 1, nan_count, nan_indices[0] ))
                sys.exit(1)
        
        # update local GPM using local data
        start = time.time()
        update_GPM(self, representation_matrix, threshold)
        end = time.time()
        

        
        print('-'*30)
        print('Representation Matrix')
        print('-'*30)
        mem = 0
        for i in range(len(representation_matrix)):
            print ('Layer {} : {}'.format(i+1,representation_matrix[i].shape))
            n, k = representation_matrix[i].shape[0], representation_matrix[i].shape[1]
            layer_memory = (n*k*32)
            mem += layer_memory

        print("representation matrix memory overhead:", mem / 8 / 1024 / 1024, "MB")
        print('-'*30)

        return end - start

    def print_final_results(self):
        print('Node {} Final Avg Accuracy: {:5.2f}%'.format(self.rank, self.acc_matrix[-1].mean()))
        bwt = np.mean((self.acc_matrix[-1] - np.diag(self.acc_matrix))[:-1])
        print('Node {} Backward Transfer: {:5.2f}%'.format(self.rank, bwt))
        
        return self.acc_matrix[-1].mean() , bwt

######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################

# Main execution
if __name__ == '__main__':
    global best_prec1
    rank = 0
    num_clients = 1
    num_clients = args.n_clients  # number of nodes
    set_seed(args.seed)
    construction_times = []
    mems = []
    gpu_allocs = []
    peak_allocs = []
    gpu_reserveds = []
    peak_reserveds = []

    if args.wandbflag:
        wandb_initialization()
        
    print(vars(args))
    
    if ((args.dataset == 'cifar100') and (args.classes == 10)) or (((args.dataset == 'cifar10') or (args.dataset == 'mnist')) and (args.classes == 100)):
        sys.exit("Dataset and classes mismatch")

    print("Number of GPU available: ", torch.cuda.device_count())
    device = torch.device("cuda:{}".format(args.device))
    val_loader, bsz_val = test_Dataset_split(args, args.num_tasks) # validation data split across the tasks. all node has same validation data

    train_loader, bsz_train, cpt = partition_trainDataset(args, rank, num_clients, device, args.num_tasks) # n-th node train_loader[n-th client][k-th task], batch size, classes per task
    client = Client(rank, num_clients, device, args, train_loader, val_loader)
    client.bsz_train = bsz_train
    client.bsz_val = bsz_val

    # Only load pretrained weights before Task 0
    # if args.arch == 'vit':
    #     load_partial_vit_weights(client.model, pretrained_name="vit_large_patch16_224")

    ###############################################
    # For each task
    start = time.time()
    for task_id in range(args.num_tasks):
        threshold = np.array([args.threshold] * client.no_layers) + task_id * np.array([args.increment_th] * client.no_layers) # threshold for GPM is increased for each task
        # For all node, initialize optimizer, scheduler
        # calculate feature matrix = M @ M^T
        client.init_task(task_id) 
                
        # For each local epoch, train the model on the local data. Only selected clients are trained
        # if task_id == 0:
        #     client.epochs = 100
        # else:
        #     client.epochs = args.local_epochs

        for epoch in range(client.epochs):
            w, avg_loss, top1 = client.train_epoch() # train on single epoch 
            client.scheduler.step()            # take gradient step 
            
            print_epoch = client.epochs // args.print_times
            if print_epoch == 0 : print_epoch = 5
            if (epoch + 1) % print_epoch == 0:
                precision, current_loss, current_acc = client.validate_seen_tasks()
                print("Epoch: {}, Task: {}, Avg train Loss: {:.4f}, Current Val Acc: {:.2f}, Accumulated Val Acc: {:.2f}".format(epoch, task_id, avg_loss, current_acc, precision))
                # print("Epoch: {}, Task: {}, Avg train Loss: {:.4f}, Validation Acc: {:.2f}%, training acc: {:.2f}%, val loss: {:.2f}".format(epoch, task_id, avg_loss, current_acc, top1, current_loss))
                
                if args.wandbflag == True:
                    wandb.log({"Task": task_id, "Epoch": epoch, "Validation Acc": precision})
            else:
                print("Epoch: {}, Task: {}, Avg train Loss: {:.4f}".format(epoch, task_id, avg_loss))
            torch.cuda.empty_cache()
        
        ###############################################
        # After finishing one task and before proceeding to next task)
        if args.gpmflag == True:
            gpm_construction_time = client.myGPM_update(threshold) # update GPM for each node: Add bases extracted from local train data of previous task
            construction_times.append(gpm_construction_time)

            process = psutil.Process(pid=os.getpid())
            mem = process.memory_info().rss / 1024 / 1024
            mems.append(mem)

            peak_alloc = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)        # torch.cuda.memory_allocated(device) / 1024 / 1024
            gpu_alloc = torch.cuda.memory_allocated(device) / 1024 / 1024
            peak_allocs.append(peak_alloc)
            gpu_allocs.append(gpu_alloc)

            gpu_reserved = torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)     # torch.cuda.memory_reserved(device) / 1024 / 1024
            peak_reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
            gpu_reserveds.append(gpu_reserved)
            peak_reserveds.append(peak_reserved)

        client.print_performance()
        accuracy_mat = client.update_acc_matrix() # update accuracy matrix
        client.print_acc_matrix()

    ###############################################
    # End of all task
    end = time.time()
    
    # Print final results of global model
    acc = []
    bwt = []
    acc_val, bwt_val = client.print_final_results()
    acc.append(acc_val)
    bwt.append(bwt_val)

    if args.gpmflag == True:
        gpm_filename = f"ViT_GPM_fp.pt"
        torch.save(client.QGPM, gpm_filename)

    
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_kb = rusage.ru_maxrss  # in kilobytes
    peak_rss_mb = peak_rss_kb / 1024.0

    print(mems)
    print(peak_allocs)
    print(gpu_allocs)
    print(peak_reserveds)
    print(gpu_reserveds)

    print("average accuracy: ", np.mean(acc) , "average bwt: ", np.mean(bwt))
    print()
    print("average gpm construction time: ", np.mean(construction_times), "total time: ", end - start)
    print()
    print("average host memory: ", np.mean(mems), "peak host memory:", peak_rss_mb, "average gpu memory allocated: ", np.mean(gpu_allocs), "average gpu memory reserved: ", np.mean(gpu_reserveds))
    print()
    print("peak gpu memory allocated: ", np.mean(peak_allocs), "peak gpu memory reserved: ", np.mean(peak_reserveds))
    print()
    print(vars(args))
    # wandb finish
    if args.wandbflag:
        wandb.finish()