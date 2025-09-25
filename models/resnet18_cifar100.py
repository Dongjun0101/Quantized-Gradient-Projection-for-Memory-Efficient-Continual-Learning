
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
import numpy as np
from copy import deepcopy
from collections import OrderedDict

__all__ = ['ResNet18_cifar100', 'get_Rmatrix_resnet18_cifar100']
##################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
import numpy as np
from copy import deepcopy
from collections import OrderedDict

def get_bucket_and_sign(col_idx, n_buckets, seed_offset=0):
    """
    Given a column index and the number of buckets, return the bucket index and sign
    for that column in a reproducible way (using a seeded RNG).
    """
    rng = np.random.RandomState(seed=col_idx + seed_offset)
    bucket = rng.randint(0, n_buckets)
    sign = rng.choice([-1, 1])
    return bucket, sign

def streaming_count_sketch_column(col_j, col_idx, n_buckets, Y, seed_offset=0):
    # Get the bucket index and sign for this column
    b, s = get_bucket_and_sign(col_idx, n_buckets, seed_offset)

    # Accumulate into the correct bucket
    # Y[:, b] += s * col_j
    Y[:, b] = Y[:, b] + s * col_j

__all__ = ['ResNet18_cifar100', 'get_Rmatrix_resnet18_cifar100']
## Define ResNet18 model
# Computes the output spatial size of a convolutional layer
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)

# Basic residual block which is a building block for ResNet
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes: # If the input and output dimensions differ (or if stride is not 1), a 1×1 convolution (with batch norm) is used to match dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.act = OrderedDict()        # Uses an ordered dictionary (self.act) to store intermediate activations under keys "conv_0" and "conv_1"
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = x      # activation 1
        self.count +=1
        out = relu(self.bn1(self.conv1(x)))             # conv1 -> BN -> ReLU
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = out    # activation 2
        self.count +=1
        out = self.bn2(self.conv2(out))                 # conv2 -> BN 
        out += self.shortcut(x)                         # Shortcut connection
        out = relu(out)                                 # ReLU                       
        return out

class ResNet_cifar100(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf):     # BasicBlock, [2, 2, 2, 2], taskcla, nf=20
        super(ResNet_cifar100, self).__init__()
        self.in_planes = nf                                 # nf determines how many channels the very first convolution produces
        self.conv1 = conv3x3(3, nf * 1, 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
        self.taskcla = taskcla
        self.linear=torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.linear.append(nn.Linear(nf * 8 * block.expansion * 4, n, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        self.act['conv_in'] = x.view(bsz, 3, 32, 32)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32)))) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        y=[]
        for t,i in self.taskcla:
            y.append(self.linear[t](out))
        return y

def ResNet18_cifar100(taskcla, nf=20):
    return ResNet_cifar100(BasicBlock, [2, 2, 2, 2], taskcla, nf)



def get_Rmatrix_resnet18_cifar100(args, net, device, data, nodes, rank, dataset): 
    # Collect activations by forward pass
    net.eval()
    
    if(data.size(dim=1)==1):
        data= data.repeat(1, 3, 1, 1)
    out  = net(data)
    
    # Activations are stored in the list act_list in the same order as defined
    act_list =[]
    act_list.extend([net.act['conv_in'], 
        net.layer1[0].act['conv_0'], net.layer1[0].act['conv_1'], net.layer1[1].act['conv_0'], net.layer1[1].act['conv_1'],
        net.layer2[0].act['conv_0'], net.layer2[0].act['conv_1'], net.layer2[1].act['conv_0'], net.layer2[1].act['conv_1'],
        net.layer3[0].act['conv_0'], net.layer3[0].act['conv_1'], net.layer3[1].act['conv_0'], net.layer3[1].act['conv_1'],
        net.layer4[0].act['conv_0'], net.layer4[0].act['conv_1'], net.layer4[1].act['conv_0'], net.layer4[1].act['conv_1']])

    batch_list  = [10,10,10,10,10,10,10,10, 50,50,50, 100,100,100,100,100,100] #scaled
    # network arch 
    stride_list = [1, 1,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1]
    map_list    = [32, 32,32,32,32, 32,16,16,16, 16,8,8,8, 8,4,4,4] 
    in_channel  = [ 3, 20,20,20,20, 20,40,40,40, 40,80,80,80, 80,160,160,160]  # expected number of channels in each corresponding activation
    # in_channel  = [ 3, 40,40,40,40, 40,80,80,80, 80,160,160,160, 160,320,320,320] # for the case of nf=40
    
    pad = 1
    sc_list=[5,9,13]        # indicate which layers will also have “shortcut” representation matrices extracted
    p1d = (1, 1, 1, 1)
    mat_final=[]            # list containing GPM Matrices 
    mat_list=[]
    mat_sc_list=[]
    for i in range(len(stride_list)):
        ksz = 3                 # patch size
        bsz=batch_list[i]
        st = stride_list[i]     
        k=0
        s=compute_conv_output_size(map_list[i],ksz,stride_list[i],pad)
        mat = np.zeros((ksz*ksz*in_channel[i],s*s*bsz))
        Y = np.zeros((ksz*ksz*in_channel[i],ksz*ksz*in_channel[i]))
        col_index = 0
        act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()
        
        for kk in range(bsz):
            for ii in range(s):
                for jj in range(s):
                    # mat[:,k]=act[kk,:,st*ii:ksz+st*ii,st*jj:ksz+st*jj].reshape(-1)
                    column = act[kk, : , st*ii : ksz + st*ii, st*jj : ksz + st*jj].reshape(-1)
                    if args.projection == True:
                        streaming_count_sketch_column(column, col_index, ksz*ksz*in_channel[i], Y)
                    else:
                        mat[:, col_index] = column
                    col_index += 1
        mat_list.append(Y)
        # For Shortcut Connection
        if i in sc_list:
            k=0
            s=compute_conv_output_size(map_list[i],1,stride_list[i])
            mat = np.zeros((1*1*in_channel[i],s*s*bsz))
            Y = np.zeros((1*1*in_channel[i],1*1*in_channel[i]))
            col_index = 0
            act = act_list[i].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        # mat[:,k]=act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].reshape(-1)
                        column = act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].reshape(-1)
                        if args.projection == True:
                            streaming_count_sketch_column(column, col_index, 1*1*in_channel[i], Y)
                        else:
                            mat[:, col_index] = column
                        col_index += 1
            mat_sc_list.append(Y) 

    ik=0
    for i in range (len(mat_list)):
        mat_final.append(mat_list[i])
        if i in [6,10,14]:
            mat_final.append(mat_sc_list[ik])
            ik+=1

    if rank == 0:
        print('-'*30)
        print('Representation Matrix')
        print('-'*30)
        for i in range(len(mat_final)):
            print ('Layer {} : {}'.format(i+1,mat_final[i].shape))
        print('-'*30)
    return mat_final 





def print_model_size(model):
    param_size, param_count = 0, 0
    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()  # in bytes

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    total_size = param_size + buffer_size
    
    # 천 단위 콤마를 적용해 출력
    print(f"Model Parameters      : {param_count:,}")
    print(f"Parameter Size       : {param_size:,} bytes")
    print(f"Buffer Size          : {buffer_size:,} bytes")
    print(f"Total Model Size     : {total_size:,} bytes "
          f"({total_size / 1024**2:,.2f} MB)")


def print_gpm_sizes(mat_final):
    """
    Given the list of representation matrices from get_Rmatrix_resnet18,
    compute and print the GPM dimension and size for each layer, then
    print the total GPM size.
    """
    total_gpm_size = 0
    for idx, mat in enumerate(mat_final, start=1):
        # 'mat' has shape (#features, #samples)
        feature_dim = mat.shape[0]
        gpm_size = feature_dim * feature_dim
        total_gpm_size += gpm_size
        print(f"Layer {idx}: Representation shape = {mat.shape}, "
              f"GPM dimension = {feature_dim}, GPM size = {gpm_size}")
    print("-" * 50)
    print(f"Total GPM size across all layers = {total_gpm_size}\n")

    
if __name__ == "__main__":
    # Example 'taskcla' with 10 tasks, each having 1 class for simplicity
    taskcla = [(t, 1) for t in range(10)]
    
    # Build the model
    net = ResNet18_cifar100(taskcla, nf=20)
    
    # Print parameter count and memory usage
    print_model_size(net)