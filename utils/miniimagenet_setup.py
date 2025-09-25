#Modified from https://github.com/LYang-666/TRGP/tree/main/dataloader

from __future__ import print_function
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

# class MiniImageNet(data.Dataset):
#     def __init__(self, root, train, transform=None):
#         super().__init__()
#         self.transform = transform
#         self.name = 'train' if train else 'test'

#         with open(os.path.join(root, f'{self.name}.pkl'), 'rb') as f:
#             data_dict = pickle.load(f)

#         self.data = data_dict['image_data']  # a NumPy array of shape (N, 84, 84, 3) presumably

#         class_dict = data_dict['class_dict']
#         class_keys = sorted(class_dict.keys())
#         class_to_int = {k: i for i, k in enumerate(class_keys)}

#         N = len(self.data)
#         self.labels = np.empty(N, dtype=np.int64)
#         for class_id_str, idxs in class_dict.items():
#             label_id = class_to_int[class_id_str]
#             for idx in idxs:
#                 self.labels[idx] = label_id

#         # If no transform is provided, at least convert PIL -> Tensor
#         if self.transform is None:
#             self.transform = transforms.Compose([
#                 transforms.ToTensor()  # Convert PIL image to Tensor
#             ])

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         # Convert the i-th image array to a PIL Image
#         img_array = self.data[i].astype('uint8')
#         img = Image.fromarray(img_array)

#         # Apply transforms => results in a Tensor, not a PIL Image
#         img = self.transform(img)

#         label = self.labels[i]
#         return img, label


# if __name__ == "__main__":
#     from torch.utils.data import DataLoader

#     dataset = MiniImageNet(root='miniimagenet', train=False)
#     loader = DataLoader(dataset, batch_size=4, shuffle=False)

#     for images, labels in loader:
#         print(images.shape, labels)
#         break





class MiniImageNet(torch.utils.data.Dataset):

    def __init__(self, root, train, transform=None):
        super(MiniImageNet, self).__init__()
        self.transform = transform
        if train:
            self.name='train'
        else:
            self.name='test'
        with open(os.path.join(root,'{}.pkl'.format(self.name)), 'rb') as f:
            data_dict = pickle.load(f)

        data = data_dict['images']
        print(type(data))
        if isinstance(data, dict):
            print("Keys in data:", data.keys())
            
        self.data = data_dict['images']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i], self.labels[i]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# dataset= MiniImageNet(root='../miniimagenet', train=True, transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize(mean,std)]))
# len = dataset.__len__()
# print(len)