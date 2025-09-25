
import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
from collections import OrderedDict

import numpy as np
import random
import pdb
import argparse,time
import math
import copy
from copy import deepcopy

from torch.distributions.uniform import Uniform


__all__ = ['alexnet', 'get_representation_matrix']

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

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def random_row_generator(i, m):
    # Create a new random state seeded by i
    rng = np.random.RandomState(seed=i)
    return rng.randn(m)

class AlexNet(nn.Module):
    def __init__(self,taskcla):
        super(AlexNet, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s=compute_conv_output_size(32,4)
        s=s//2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256*self.smid*self.smid,2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = nn.Linear(2048,2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])
        
        self.taskcla = taskcla

        self.fc3=torch.nn.ModuleList()
        for t,n in self.taskcla: # taskcla = [(0,1), (1,1), (2,1), (3,1), (4,1), (5,1), (6,1), (7,1), (8,1), (9,1)] >> t = task, n = num of classes / num of tasks
            self.fc3.append(torch.nn.Linear(2048,n,bias=False)) # appends a new fully connected layer (torch.nn.Linear with input feature of 2048 and output feature of n) to the ModuleList for each task
        
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1']=x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2']=x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3']=x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x=x.view(bsz,-1)
        self.act['fc1']=x
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        self.act['fc2']=x        
        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))
        y=[]

        for t,i in self.taskcla: # select fc3 for corresponding task(train the last layer from scratch)
            y.append(self.fc3[t](x))
            
        return y



def alexnet(taskcla):
    return AlexNet(taskcla)

def get_representation_matrix (net, device, data, nodes, rank, args): 
    # Collect activations by forward pass
    n = True
    net.eval()
    out  = net(data)
    batch_list=[24,100,100,125,125] #default case

    mat_list=[]
    act_key=list(net.act.keys())
    for i in range(len(net.map)):
        bsz=batch_list[i]
        k=0
        if i<3:     # for convolution layers
            ksz= net.ksize[i]
            s=compute_conv_output_size(net.map[i],net.ksize[i])
            nn = net.ksize[i]*net.ksize[i]*net.in_channel[i]
            kk = s*s*bsz
            mat = np.zeros((nn,kk))
            Y = np.zeros((nn,nn))
            col_index = 0
            # mat = np.zeros((net.ksize[i]*net.ksize[i]*net.in_channel[i],  net.ksize[i]*net.ksize[i]*net.in_channel[i]))
            act = net.act[act_key[i]].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        column = act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) # mat[:,k] = k-th column of i-th layer's activation matrix
                        
                        # # (i) Gaussian random projection
                        # np.random.seed(col_index)
                        # random_vector = np.random.randn(nn)
                        # Y = Y + np.outer(column, random_vector)
                        # col_index += 1
                        
                        # # (ii) Sparse random projection (count sketch)
                        if args.projection == True:
                            streaming_count_sketch_column(column, col_index, nn, Y)
                            col_index += 1
                        else:
                            mat[:, col_index] = column
                            col_index += 1
                        
            if args.projection == True:
                mat_list.append(Y)    # mat = i-th layer's activation matrix
            else:
                mat_list.append(mat)    # mat = i-th layer's activation matrix
            
        else:       # for FC layers
            act = net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)

    if(rank==0):
        print('-'*30)
        print('Representation Matrix')
        print('-'*30)
        for i in range(len(mat_list)):
            print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
        print('-'*30)
        n = False
    return mat_list    

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

    
if __name__ == "__main__":
    # Example 'taskcla' with 10 tasks, each having 1 class for simplicity
    taskcla = [(t, 1) for t in range(10)]
    
    # Build the model
    net = alexnet(taskcla)
    
    # Print parameter count and memory usage
    print_model_size(net)