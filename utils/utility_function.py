import argparse
import os
import time
import numpy as np
import copy
import sys
import random
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
# from scipy.linalg import svd  # Use SciPy's SVD function
# from scipy.stats import norm
# from scipy.stats import kurtosis
# from scipy.stats import shapiro
from PIL import Image
from torch.utils.data import Dataset
from collections import OrderedDict
from copy import deepcopy

# Importing modules related to your specific implementations
from models import *
import wandb

# --- HF DomainNet adapter ---
import os
from typing import List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Set custom cache directory for Hugging Face datasets
CACHE_DIR = "/scratch/10327/dongjunkim0101/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

try:
    from datasets import load_dataset, Dataset as HFDataset
except Exception as e:
    load_dataset = None  # we'll assert later

# short -> long domain names (for convenience)
DOMAINNET_SHORT2LONG = {
    'c': 'clipart',
    'i': 'infograph',
    'p': 'painting',
    'q': 'quickdraw',
    'r': 'real',
    's': 'sketch',
}
DOMAINNET_ALL = ['c','i','p','q','r','s']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_domainnet_train_tf(image_size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def build_domainnet_eval_tf(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def _domain_ids_from_hf(hf_split, requested: List[str]):
    """Map ['c','i',...] or long names to the integer ids in HF features."""
    feat = hf_split.features["domain"]
    names = list(feat.names)  # e.g., ['clipart','infograph','painting','quickdraw','real','sketch']
    name2id = {n:i for i,n in enumerate(names)}
    ids = []
    for d in requested:
        if d in DOMAINNET_SHORT2LONG:
            d = DOMAINNET_SHORT2LONG[d]
        if d not in name2id:
            raise KeyError(f"Requested domain '{d}' is not in HF dataset domains: {names}")
        ids.append(name2id[d])
    return ids, names  # (list of ints, full ordered names)

class HFDomainNetTorch(Dataset):
    """
    Wrap an HF split (datasets.Dataset) and yield (image_tensor, label_int).
    Optionally filter to a single domain_id.
    """
    def __init__(self, hf_split: "HFDataset", transform=None, domain_id: Optional[int]=None, num_proc: Optional[int]=None):
        if domain_id is not None:
            # keep only rows with that domain_id
            # (batched=False to avoid materializing huge batches)
            hf_split = hf_split.filter(lambda ex: ex["domain"] == domain_id, num_proc=num_proc)
        self.hf = hf_split
        self.transform = transform

    def __len__(self):
        return self.hf.num_rows

    def __getitem__(self, idx: int):
        ex = self.hf[idx]
        img = ex["image"]  # PIL.Image
        if not isinstance(img, Image.Image):
            # safety: some environments may decode to array; convert to PIL
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(ex["label"])


class MinimalMultiDatasetPartitioner:
    """
    Shard each dataset in `datasets_list` across ranks with simple striding.
    Provides .use(rank, n) -> sharded Dataset, matching your current API.
    """
    def __init__(self, datasets_list: List[Dataset], size: int):
        self.datasets_list = datasets_list
        self.size = max(1, int(size))

    def use(self, rank: int, n: int) -> Dataset:
        base = self.datasets_list[n]
        idxs = list(range(len(base)))
        shard = idxs[rank::self.size]
        # use your existing Partition if defined; else fallback
        try:
            return Partition(base, shard)
        except NameError:
            class _Partition(Dataset):
                def __init__(self, base, indices): self.base, self.indices = base, indices
                def __len__(self): return len(self.indices)
                def __getitem__(self, i): return self.base[self.indices[i]]
            return _Partition(base, shard)




def format_val(x):
    """
    If x is really an integer (type int or a float with x.is_integer()==True),
    print an integer. Otherwise, print with 4 decimal places.
    """
    # If the object is actually an int
    if isinstance(x, int):
        return str(x)
    # If it's a float but effectively an integer, e.g. 3.0
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    # Otherwise, treat it like a float with 4 decimal digits
    return f"{float(x):.4f}"

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

def skew_sort(indices, skew, classes, class_size, seed):
    # skew belongs to [0,1]
    rng = Random()
    rng.seed(seed)
    class_indices = {}
    for i in range(0, classes):
        class_indices[i] = indices[0:class_size[i]]
        indices = indices[class_size[i]:]
    random_indices = []
    sorted_indices = []
    for i in range(0, classes):
        sorted_size = int(skew * class_size[i])
        sorted_indices = sorted_indices + class_indices[i][0:sorted_size]
        random_indices = random_indices + class_indices[i][sorted_size:]
    rng.shuffle(random_indices)
    return random_indices, sorted_indices

class DataPartitioner(object):
    """ Partitions a dataset into different chunks"""

    def __init__(self, data, sizes, skew, classes, class_size, seed, device, tasks=2):
        assert classes % tasks == 0
        self.data = data
        self.partitions = {}
        data_len = len(data)
        dataset = torch.utils.data.DataLoader(data, batch_size=512, shuffle=False, num_workers=8)
        labels = []
        for batch_idx, (inputs, targets) in enumerate(dataset):
            labels = labels + targets.tolist()
        sort_index = np.argsort(np.array(labels))
        indices_full = sort_index.tolist()
        task_data_len = int(data_len / tasks)

        for n in range(tasks):
            ind_per_task = indices_full[n * task_data_len: (n + 1) * task_data_len]
            indices_rand, indices = skew_sort(ind_per_task, skew=skew, classes=int(classes / tasks),
                                              class_size=class_size, seed=seed)
            self.partitions[n] = []
            for frac in sizes:
                if skew == 1:  # completely non iid
                    part_len = int(frac * task_data_len)
                    self.partitions[n].append(indices[0:part_len])
                    indices = indices[part_len:]
                elif skew == 0:  # iid setting
                    part_len = int(frac * task_data_len)
                    self.partitions[n].append(indices_rand[0:part_len])
                    indices_rand = indices_rand[part_len:]  # remove to use full data at each node for experiment
                else:  # partially non-iid
                    part_len = int(frac * task_data_len * skew);
                    part_len_rand = int(frac * task_data_len * (1 - skew))
                    part_ind = indices[0:part_len] + indices_rand[0:part_len_rand]
                    self.partitions[n].append(part_ind)
                    indices = indices[part_len:]
                    indices_rand = indices_rand[part_len_rand:]

    def use(self, partition, task):
        return Partition(self.data, self.partitions[task][partition])

class DataPartition_5set(object):
    """ Partitions 5-datasets across different nodes, not setup for non-IID data yet, works only for SKEW=0"""
    def __init__(self, data_type, data, sizes, skew, classes, class_size, seed, device, tasks=2):
        #assert classes%tasks==0
        self.data = data
        self.partitions = {}
        indices_full = []
        data_len= []

        for i in range(len(data)):
            dataset = torch.utils.data.DataLoader(data[i], batch_size=512, shuffle=False, num_workers=2)
            data_len.append(len(data[i]))
            labels= []

            if(data_type=='5datasets'):
                for batch_idx, (inputs, targets) in enumerate(dataset):
                    labels = labels+targets.tolist()
            else:
                for batch_idx, (inputs, targets) in enumerate(dataset):
                    t = np.array(targets.tolist()).reshape(-1)
                    labels = labels+t.tolist()

            sort_index = np.argsort(np.array(labels))
            indices_full.append(sort_index.tolist())

        for n in range(tasks):
            task_data_len = int(data_len[n])
            ind_per_task = indices_full[n]
            rng = Random()
            rng.seed(seed)
            rng.shuffle(ind_per_task)
            self.partitions[n] = []
            for frac in sizes:
                part_len = int(frac*task_data_len)
                self.partitions[n].append(ind_per_task[0:part_len])
                ind_per_task = ind_per_task[part_len:] #remove to use full data at each node for experiment

    def use(self, partition, task):
        return Partition(self.data[task], self.partitions[task][partition])

def partition_trainDataset(args, rank, size, device, tasks=2):
    """Partitioning dataset"""
    if args.dataset == 'mnist':
        normalize = transforms.Normalize((0.1307,), (0.3081,))  # mnist have one channel only
        classes = 10
        class_size = {x: 6000 for x in range(10)}  # each class has 6000 sample images
        dataset = datasets.MNIST(root=f'data_mnist', train=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
        c = int(classes / tasks)  # 10 classes / tasks
    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        classes = 10
        class_size = {x: 5000 for x in range(10)}  # creates a dictionary where each class (from 0 to 9) is assigned a size of 5000.
        dataset = datasets.CIFAR10(root=f'data_cifar10', train=True, transform=transforms.Compose([  # dataset loading
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
        c = int(classes / tasks)  # 10 classes / tasks
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        classes = 100
        class_size = {x: 500 for x in range(100)}
        c = int(classes / tasks)  # 100 classes / tasks
        dataset = datasets.CIFAR100(root=f'../data/data_cifar100', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    elif args.dataset == '5datasets':
        dataset= []
        classes= 10 #each task has 10 classes
        c= int(classes)

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]
        dataset_1= datasets.CIFAR10(root=f'../data/Five_data/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.1,) 
        std=(0.2752,)
        dataset_2= datasets.MNIST(root=f'../data/Five_data/',train=True,download=True,transform=transforms.Compose([transforms.Pad(padding=2,fill=0),transforms.ToTensor(), transforms.Normalize(mean,std)]))
        
        mean=[0.4377,0.4438,0.4728]
        std=[0.198,0.201,0.197]
        dataset_3= datasets.SVHN(root=f'../data/Five_data/SVHN',split='train',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.2190,)
        std=(0.3318,)
        dataset_4= datasets.FashionMNIST(root=f'../data/Five_data/', train=True, download=True, transform=transforms.Compose([
                    transforms.Pad(padding=2, fill=0), transforms.ToTensor(),transforms.Normalize(mean, std)]))

        mean=(0.4254,)
        std=(0.4501,)
        dataset_5= notmnist_setup.notMNIST(root=f'../data/Five_data/notmnist', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        dataset= [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]
    elif args.dataset == 'miniimagenet':
        dataset= []
        classes= 100 #each task has 5 classes
        c = int(classes/tasks)
        class_size = {x:500 for x in range(100)} 
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        dataset= miniimagenet_setup.MiniImageNet(root='../data/miniimagenet', train=True, transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean,std)]))
    elif args.dataset in ['domainnet_hf', 'domainnet-hf']:
        assert load_dataset is not None, "Please `pip install datasets`"

        # config
        image_size = getattr(args, 'domainnet_img_size', 224)
        domains = getattr(args, 'domainnet_domains', DOMAINNET_ALL)
        # limit number of tasks to number of domains chosen
        tasks = min(tasks, len(domains))

        # load HF dataset (downloads/caches under custom cache directory)
        print(f"Loading DomainNet dataset with cache directory: {CACHE_DIR}")
        try:
            hf = load_dataset("wltjr1007/DomainNet", cache_dir=CACHE_DIR)  # splits: 'train', 'test'  (409,832 / 176,743)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Trying to clear cache and retry...")
            import shutil
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)
            hf = load_dataset("wltjr1007/DomainNet", cache_dir=CACHE_DIR)
        
        hf_train = hf["train"]

        # resolve requested domain ids (integers) in this HF dataset
        dom_ids, dom_names = _domain_ids_from_hf(hf_train, domains)

        # transforms
        train_tf = build_domainnet_train_tf(image_size=image_size)

        # build one torch Dataset per domain (task)
        dataset_list = []
        for d_id in dom_ids[:tasks]:
            dataset_list.append(HFDomainNetTorch(hf_train, transform=train_tf, domain_id=d_id))

        # your loader/partition boilerplate
        train_set = {}
        bsz = int((args.batch_size) / float(size))
        partition_sizes = [1.0 / size for _ in range(size)]
        if rank == 0:
            print("Data partition_sizes among clients:", partition_sizes)
            print("HF DomainNet domains:", [dom_names[i] for i in dom_ids[:tasks]])

        # shard each domain dataset across ranks
        partition = MinimalMultiDatasetPartitioner(dataset_list, size=size)

        # classes per task: DomainNet label space has 345 classes (shared across domains)
        c = 345

        for n in range(tasks):
            task_partition = partition.use(rank, n)
            train_set[n] = torch.utils.data.DataLoader(
                task_partition,
                batch_size=bsz,
                shuffle=True,
                num_workers=8,
                drop_last=(size == 8)
            )

        return train_set, bsz, c
        
        
    train_set = {}

    bsz = int((args.batch_size) / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]  # For 4 nodes, partition_size = [0.25, 0.25, 0.25, 0.25]

    if (rank == 0):
        print("Data partition_sizes among clients:", partition_sizes)

    # partition object creation
    # normalized entire dataset, partition size among the nodes, skewness(0.0 for iid, 1.0 for non-iid), classes, class size, seed, device, tasks
    if(args.dataset=='5datasets'):
        partition= DataPartition_5set(args.dataset, dataset, partition_sizes, skew=args.skew, classes=classes, class_size=0, seed=args.seed, device=device, tasks=tasks)
    else:
        partition = DataPartitioner(dataset, partition_sizes, skew=args.skew, classes=classes, class_size=class_size, seed=args.seed, device=device, tasks=tasks)
    
    # partition for continual learning
    for n in range(tasks):
        task_partition = partition.use(rank, n)  # partition for corresponding rank(node) / n-th task
        train_set[n] = torch.utils.data.DataLoader(task_partition, batch_size=bsz, shuffle=True, num_workers=8, drop_last=(size == 8))  # train_set for n-th task

    return train_set, bsz, c # c = class per task

def test_Dataset_split(args, tasks):
    if args.dataset == 'mnist':
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        dataset = datasets.MNIST(root=f'data_mnist', train=False, download=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ]))
    elif args.dataset == 'cifar10':

        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        dataset = datasets.CIFAR10(root=f'data_cifar10', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        dataset = datasets.CIFAR100(root=f'../data/data_cifar100', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    elif args.dataset == '5datasets':
        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]
        dataset_1= datasets.CIFAR10(root=f'../data/Five_data/',train=False, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.1,)
        std=(0.2752,)
        dataset_2= datasets.MNIST(root=f'../data/Five_data/',train=False, download=True,transform=transforms.Compose([transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std)]))

        loader = torch.utils.data.DataLoader(dataset_2, batch_size=1, shuffle=False)
        for image, target in loader:
            image=image.expand(1,3,image.size(2),image.size(3))

        mean=[0.4377,0.4438,0.4728]
        std=[0.198,0.201,0.197]
        dataset_3= datasets.SVHN(root=f'../data/Five_data/SVHN',split='test',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.2190,)
        std=(0.3318,)
        dataset_4= datasets.FashionMNIST(root=f'../data/Five_data/', train=False, download=True, transform=transforms.Compose([
                    transforms.Pad(padding=2, fill=0), transforms.ToTensor(),transforms.Normalize(mean, std)]))
        mean=(0.4254,)
        std=(0.4501,)
        dataset_5= notmnist_setup.notMNIST(root=f'../data/Five_data/notmnist', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        dataset= [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]

        val_set={}
        val_bsz = 64

        for n in range(tasks):
            task_data = dataset[n]
            val_set[n] = torch.utils.data.DataLoader(task_data, batch_size=val_bsz, shuffle=True, num_workers=5) #shuffle=False gives low test acc for bn with track_run_stats=False
    elif args.dataset == 'miniimagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        dataset= miniimagenet_setup.MiniImageNet(root='../data/miniimagenet', train=False, transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean,std)]))

        data_len = len(dataset)
        d = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=1)
        labels = []
        for batch_idx, (inputs, targets) in enumerate(d):
            labels = labels+targets.tolist()

        sort_index = np.argsort(np.array(labels))
        indices = sort_index.tolist()
        task_data_len = int(data_len/tasks)
        val_set={}
        val_bsz=10

        for n in range(tasks):
            ind_per_task = indices[n*task_data_len: (n+1)*task_data_len]
            task_data = Partition(dataset, ind_per_task)
            val_set[n] = torch.utils.data.DataLoader(task_data, batch_size=val_bsz, shuffle=True, num_workers=2)
    elif args.dataset in ['domainnet_hf', 'domainnet-hf']:
        assert load_dataset is not None, "Please `pip install datasets`"

        image_size = getattr(args, 'domainnet_img_size', 224)
        domains = getattr(args, 'domainnet_domains', DOMAINNET_ALL)
        tasks = min(tasks, len(domains))

        print(f"Loading DomainNet test dataset with cache directory: {CACHE_DIR}")
        try:
            hf = load_dataset("wltjr1007/DomainNet", cache_dir=CACHE_DIR)
        except Exception as e:
            print(f"Error loading test dataset: {e}")
            print("Trying to clear cache and retry...")
            import shutil
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)
            hf = load_dataset("wltjr1007/DomainNet", cache_dir=CACHE_DIR)
        
        hf_test = hf["test"]  # 176,743 examples in the split
        dom_ids, dom_names = _domain_ids_from_hf(hf_test, domains)

        eval_tf = build_domainnet_eval_tf(image_size=image_size)
        val_set = {}
        val_bsz = 64

        for n, d_id in enumerate(dom_ids[:tasks]):
            dset = HFDomainNetTorch(hf_test, transform=eval_tf, domain_id=d_id)
            val_set[n] = torch.utils.data.DataLoader(dset, batch_size=val_bsz, shuffle=True, num_workers=5)

        # important: return early so generic label-splitter doesn't override this
        return val_set, val_bsz   
     
    if (args.dataset != '5datasets') and (args.dataset != 'miniimagenet'):
        val_set = {}
        data_len = len(dataset)
        d = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=5)
        labels = []
        for batch_idx, (inputs, targets) in enumerate(d):
            labels = labels + targets.tolist()
        sort_index = np.argsort(np.array(labels))
        indices = sort_index.tolist()
        task_data_len = int(data_len / tasks)
        val_bsz = 64

        for n in range(tasks):
            ind_per_task = indices[n * task_data_len: (n + 1) * task_data_len]
            task_data = Partition(dataset, ind_per_task)
            val_set[n] = torch.utils.data.DataLoader(task_data, batch_size=val_bsz, shuffle=True, num_workers=5)

    return val_set, val_bsz

def average_weights(w): # w : list of weights, weight = node.model.state_dict()
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


import argparse
import os
import time
import numpy as np
import copy
import sys
import random
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
from scipy.linalg import svd  # Use SciPy's SVD function
from scipy.stats import norm
from scipy.stats import kurtosis
from scipy.stats import shapiro
from PIL import Image
from torch.utils.data import Dataset
from collections import OrderedDict
from copy import deepcopy

# Importing modules related to your specific implementations
from models import *
import wandb
from collections import Counter
from torchvision import datasets, transforms

from collections import defaultdict

def _labels_from_imagefolder(ds):
    # Works across torchvision versions
    if hasattr(ds, "targets") and ds.targets:
        return list(ds.targets)
    if hasattr(ds, "samples") and ds.samples:
        return [y for _, y in ds.samples]
    if hasattr(ds, "imgs") and ds.imgs:
        return [y for _, y in ds.imgs]
    raise AttributeError("Cannot extract labels from ImageFolder dataset.")

def class_sizes_from_imagefolder(ds):
    """Return {class_index: count} for ImageFolder, regardless of torchvision version."""
    if hasattr(ds, "targets") and ds.targets:
        labels = ds.targets
    elif hasattr(ds, "samples") and ds.samples:
        labels = [y for _, y in ds.samples]
    elif hasattr(ds, "imgs") and ds.imgs:
        labels = [y for _, y in ds.imgs]
    else:
        raise AttributeError("Cannot find labels in ImageFolder dataset.")
    cnt = Counter(labels)
    return {i: cnt.get(i, 0) for i in range(len(ds.classes))}

def format_val(x):
    """
    If x is really an integer (type int or a float with x.is_integer()==True),
    print an integer. Otherwise, print with 4 decimal places.
    """
    # If the object is actually an int
    if isinstance(x, int):
        return str(x)
    # If it's a float but effectively an integer, e.g. 3.0
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    # Otherwise, treat it like a float with 4 decimal digits
    return f"{float(x):.4f}"

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

def skew_sort(indices, skew, classes, class_size, seed):
    # skew belongs to [0,1]
    rng = Random()
    rng.seed(seed)
    class_indices = {}
    for i in range(0, classes):
        class_indices[i] = indices[0:class_size[i]]
        indices = indices[class_size[i]:]
    random_indices = []
    sorted_indices = []
    for i in range(0, classes):
        sorted_size = int(skew * class_size[i])
        sorted_indices = sorted_indices + class_indices[i][0:sorted_size]
        random_indices = random_indices + class_indices[i][sorted_size:]
    rng.shuffle(random_indices)
    return random_indices, sorted_indices

class DataPartitioner(object):
    """ Partitions a dataset into different chunks"""

    def __init__(self, data, sizes, skew, classes, class_size, seed, device, tasks=2):
        assert classes % tasks == 0
        self.data = data
        self.partitions = {}
        data_len = len(data)
        dataset = torch.utils.data.DataLoader(data, batch_size=512, shuffle=False, num_workers=8)
        labels = []
        for batch_idx, (inputs, targets) in enumerate(dataset):
            labels = labels + targets.tolist()
        sort_index = np.argsort(np.array(labels))
        indices_full = sort_index.tolist()
        task_data_len = int(data_len / tasks)

        for n in range(tasks):
            ind_per_task = indices_full[n * task_data_len: (n + 1) * task_data_len]
            indices_rand, indices = skew_sort(ind_per_task, skew=skew, classes=int(classes / tasks),
                                              class_size=class_size, seed=seed)
            self.partitions[n] = []
            for frac in sizes:
                if skew == 1:  # completely non iid
                    part_len = int(frac * task_data_len)
                    self.partitions[n].append(indices[0:part_len])
                    indices = indices[part_len:]
                elif skew == 0:  # iid setting
                    part_len = int(frac * task_data_len)
                    self.partitions[n].append(indices_rand[0:part_len])
                    indices_rand = indices_rand[part_len:]  # remove to use full data at each node for experiment
                else:  # partially non-iid
                    part_len = int(frac * task_data_len * skew);
                    part_len_rand = int(frac * task_data_len * (1 - skew))
                    part_ind = indices[0:part_len] + indices_rand[0:part_len_rand]
                    self.partitions[n].append(part_ind)
                    indices = indices[part_len:]
                    indices_rand = indices_rand[part_len_rand:]

    def use(self, partition, task):
        return Partition(self.data, self.partitions[task][partition])

class DataPartition_5set(object):
    """ Partitions 5-datasets across different nodes, not setup for non-IID data yet, works only for SKEW=0"""
    def __init__(self, data_type, data, sizes, skew, classes, class_size, seed, device, tasks=2):
        #assert classes%tasks==0
        self.data = data
        self.partitions = {}
        indices_full = []
        data_len= []

        for i in range(len(data)):
            dataset = torch.utils.data.DataLoader(data[i], batch_size=512, shuffle=False, num_workers=2)
            data_len.append(len(data[i]))
            labels= []

            if(data_type=='5datasets'):
                for batch_idx, (inputs, targets) in enumerate(dataset):
                    labels = labels+targets.tolist()
            else:
                for batch_idx, (inputs, targets) in enumerate(dataset):
                    t = np.array(targets.tolist()).reshape(-1)
                    labels = labels+t.tolist()

            sort_index = np.argsort(np.array(labels))
            indices_full.append(sort_index.tolist())

        for n in range(tasks):
            task_data_len = int(data_len[n])
            ind_per_task = indices_full[n]
            rng = Random()
            rng.seed(seed)
            rng.shuffle(ind_per_task)
            self.partitions[n] = []
            for frac in sizes:
                part_len = int(frac*task_data_len)
                self.partitions[n].append(ind_per_task[0:part_len])
                ind_per_task = ind_per_task[part_len:] #remove to use full data at each node for experiment

    def use(self, partition, task):
        return Partition(self.data[task], self.partitions[task][partition])

def partition_trainDataset(args, rank, size, device, tasks=2):
    """Partitioning dataset"""
    if args.dataset == 'mnist':
        normalize = transforms.Normalize((0.1307,), (0.3081,))  # mnist have one channel only
        classes = 10
        class_size = {x: 6000 for x in range(10)}  # each class has 6000 sample images
        dataset = datasets.MNIST(root=f'data_mnist', train=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
        c = int(classes / tasks)  # 10 classes / tasks
    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        classes = 10
        class_size = {x: 5000 for x in range(10)}  # creates a dictionary where each class (from 0 to 9) is assigned a size of 5000.
        dataset = datasets.CIFAR10(root=f'data_cifar10', train=True, transform=transforms.Compose([  # dataset loading
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
        c = int(classes / tasks)  # 10 classes / tasks
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        classes = 100
        class_size = {x: 500 for x in range(100)}
        c = int(classes / tasks)  # 100 classes / tasks
        dataset = datasets.CIFAR100(root=f'../data/data_cifar100', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    elif args.dataset == '5datasets':
        dataset= []
        classes= 10 #each task has 10 classes
        c= int(classes)

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]
        dataset_1= datasets.CIFAR10(root=f'../data/Five_data/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.1,) 
        std=(0.2752,)
        dataset_2= datasets.MNIST(root=f'../data/Five_data/',train=True,download=True,transform=transforms.Compose([transforms.Pad(padding=2,fill=0),transforms.ToTensor(), transforms.Normalize(mean,std)]))
        
        mean=[0.4377,0.4438,0.4728]
        std=[0.198,0.201,0.197]
        dataset_3= datasets.SVHN(root=f'../data/Five_data/SVHN',split='train',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.2190,)
        std=(0.3318,)
        dataset_4= datasets.FashionMNIST(root=f'../data/Five_data/', train=True, download=True, transform=transforms.Compose([
                    transforms.Pad(padding=2, fill=0), transforms.ToTensor(),transforms.Normalize(mean, std)]))

        mean=(0.4254,)
        std=(0.4501,)
        dataset_5= notmnist_setup.notMNIST(root=f'../data/Five_data/notmnist', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        dataset= [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]
    elif args.dataset == 'miniimagenet':
        dataset= []
        classes= 100 #each task has 5 classes
        c = int(classes/tasks)
        class_size = {x:500 for x in range(100)} 
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        dataset= miniimagenet_setup.MiniImageNet(root='../data/miniimagenet', train=True, transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean,std)]))
    elif args.dataset in ['imagenet_r', 'imagenet-r', 'imagenetr']:
        # Standard ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # allow override via flag like --imagenet_r_root
        root = getattr(args, 'imagenet_r_root', '../data/imagenet_r')
        train_dir = os.path.join(root, 'train')

        dataset = datasets.ImageFolder(train_dir, transform=train_tf)
        classes = len(dataset.classes)           # should be 200 on ImageNet-R
        class_size = class_sizes_from_imagefolder(dataset)

        # classes per task (adjust if tasks doesn't evenly divide 200)
        c = int(classes / tasks)
        
        
    train_set = {}

    bsz = int((args.batch_size) / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]  # For 4 nodes, partition_size = [0.25, 0.25, 0.25, 0.25]

    if (rank == 0):
        print("Data partition_sizes among clients:", partition_sizes)

    # partition object creation
    # normalized entire dataset, partition size among the nodes, skewness(0.0 for iid, 1.0 for non-iid), classes, class size, seed, device, tasks
    if(args.dataset=='5datasets'):
        partition= DataPartition_5set(args.dataset, dataset, partition_sizes, skew=args.skew, classes=classes, class_size=0, seed=args.seed, device=device, tasks=tasks)
    else:
        partition = DataPartitioner(dataset, partition_sizes, skew=args.skew, classes=classes, class_size=class_size, seed=args.seed, device=device, tasks=tasks)
    
    # partition for continual learning
    for n in range(tasks):
        task_partition = partition.use(rank, n)  # partition for corresponding rank(node) / n-th task
        train_set[n] = torch.utils.data.DataLoader(task_partition, batch_size=bsz, shuffle=True, num_workers=8, drop_last=(size == 8))  # train_set for n-th task

    return train_set, bsz, c # c = class per task

def test_Dataset_split(args, tasks):
    if args.dataset == 'mnist':
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        dataset = datasets.MNIST(root=f'data_mnist', train=False, download=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ]))
    elif args.dataset == 'cifar10':

        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        dataset = datasets.CIFAR10(root=f'data_cifar10', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        dataset = datasets.CIFAR100(root=f'../data/data_cifar100', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    elif args.dataset == '5datasets':
        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]
        dataset_1= datasets.CIFAR10(root=f'../data/Five_data/',train=False, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.1,)
        std=(0.2752,)
        dataset_2= datasets.MNIST(root=f'../data/Five_data/',train=False, download=True,transform=transforms.Compose([transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std)]))

        loader = torch.utils.data.DataLoader(dataset_2, batch_size=1, shuffle=False)
        for image, target in loader:
            image=image.expand(1,3,image.size(2),image.size(3))

        mean=[0.4377,0.4438,0.4728]
        std=[0.198,0.201,0.197]
        dataset_3= datasets.SVHN(root=f'../data/Five_data/SVHN',split='test',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        mean=(0.2190,)
        std=(0.3318,)
        dataset_4= datasets.FashionMNIST(root=f'../data/Five_data/', train=False, download=True, transform=transforms.Compose([
                    transforms.Pad(padding=2, fill=0), transforms.ToTensor(),transforms.Normalize(mean, std)]))
        mean=(0.4254,)
        std=(0.4501,)
        dataset_5= notmnist_setup.notMNIST(root=f'../data/Five_data/notmnist', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        dataset= [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]

        val_set={}
        val_bsz = 64

        for n in range(tasks):
            task_data = dataset[n]
            val_set[n] = torch.utils.data.DataLoader(task_data, batch_size=val_bsz, shuffle=True, num_workers=5) #shuffle=False gives low test acc for bn with track_run_stats=False
    elif args.dataset == 'miniimagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        dataset= miniimagenet_setup.MiniImageNet(root='../data/miniimagenet', train=False, transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean,std)]))

        data_len = len(dataset)
        d = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=1)
        labels = []
        for batch_idx, (inputs, targets) in enumerate(d):
            labels = labels+targets.tolist()

        sort_index = np.argsort(np.array(labels))
        indices = sort_index.tolist()
        task_data_len = int(data_len/tasks)
        val_set={}
        val_bsz=10

        for n in range(tasks):
            ind_per_task = indices[n*task_data_len: (n+1)*task_data_len]
            task_data = Partition(dataset, ind_per_task)
            val_set[n] = torch.utils.data.DataLoader(task_data, batch_size=val_bsz, shuffle=True, num_workers=2)
    elif args.dataset in ['imagenet_r', 'imagenet-r', 'imagenetr']:
        # Deterministic eval transforms (ImageNet standard)
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        eval_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # Root can be overridden by --imagenet_r_root if you like (keeps symmetry with train side)
        root = getattr(args, 'imagenet_r_root', '../data/imagenet_r')
        # Prefer 'val', but support 'test' or raw 'imagenet-r' (unsplit) as fallback
        for sub in ['val', 'test', 'imagenet-r']:
            eval_dir = os.path.join(root, sub)
            if os.path.isdir(eval_dir):
                break
        else:
            raise FileNotFoundError(
                f"Could not find a valid ImageNet-R eval directory under '{root}'. "
                f"Expected one of: {os.path.join(root,'val')}, {os.path.join(root,'test')}, {os.path.join(root,'imagenet-r')}"
            )

        dataset = datasets.ImageFolder(eval_dir, transform=eval_tf)
        labels = _labels_from_imagefolder(dataset)
        num_classes = len(dataset.classes)

        # -------- Partition by whole classes per task --------
        # This avoids cutting a class across tasks (ImageNet-R is imbalanced per class).
        # If you always choose tasks that evenly divide num_classes (e.g., 200 % tasks == 0),
        # each task gets the same number of classes. Otherwise, we distribute the remainder.
        classes_per_task = num_classes // tasks
        remainder = num_classes % tasks

        # Build index lists per class
        idxs_per_class = defaultdict(list)
        for idx, y in enumerate(labels):
            idxs_per_class[y].append(idx)

        val_set = {}
        val_bsz = 64  # match your other eval loaders

        start = 0
        for n in range(tasks):
            # Distribute any remainder (+1 class) to the first 'remainder' tasks
            take = classes_per_task + (1 if n < remainder else 0)
            cls_ids = list(range(start, start + take))
            start += take

            # Flatten indices for this block of classes
            task_indices = []
            for cid in cls_ids:
                task_indices.extend(idxs_per_class[cid])

            task_data = Partition(dataset, task_indices)
            # Keep shuffle=True to match your other eval loaders
            val_set[n] = torch.utils.data.DataLoader(task_data, batch_size=val_bsz,
                                                    shuffle=True, num_workers=5)

        # Important: return here so the generic label-sorted splitter below doesn't run
        return val_set, val_bsz
     
    if (args.dataset != '5datasets') and (args.dataset != 'miniimagenet'):
        val_set = {}
        data_len = len(dataset)
        d = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=5)
        labels = []
        for batch_idx, (inputs, targets) in enumerate(d):
            labels = labels + targets.tolist()
        sort_index = np.argsort(np.array(labels))
        indices = sort_index.tolist()
        task_data_len = int(data_len / tasks)
        val_bsz = 64

        for n in range(tasks):
            ind_per_task = indices[n * task_data_len: (n + 1) * task_data_len]
            task_data = Partition(dataset, ind_per_task)
            val_set[n] = torch.utils.data.DataLoader(task_data, batch_size=val_bsz, shuffle=True, num_workers=5)

    return val_set, val_bsz

def average_weights(w): # w : list of weights, weight = node.model.state_dict()
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res