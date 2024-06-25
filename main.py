import os
import numpy as np
import scipy.misc as m
import skimage
import skimage.io
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
import dataloaders as tr
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from utils_main import *
from torch.utils.data import DataLoader,ConcatDataset
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import timeit
import metrics
from torchmetrics import CosineSimilarity
import random
from torch.autograd import Function
import warnings
from pytorch_wavelets import DWTForward, DWTInverse
warnings.filterwarnings('ignore')
import math
from torch.backends import cudnn
import imageio
from deepv3 import *
import time
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
world_size = 1
local_rank = 0

if 'WORLD_SIZE' in os.environ:
    world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(local_rank)
print('My Rank:', local_rank)
dist_url = 'tcp://127.0.0.1:' + str(8000 + (int(time.time()%1000))//10)

torch.distributed.init_process_group(backend='nccl',
                                        init_method=dist_url,
                                        world_size=world_size, rank=local_rank)

class CityscapesSegmentation(data.Dataset):
    NUM_CLASSES = 19 

    def __init__(self, args, root=Path.db_root_dir('cityscapes'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]    # for all datasets except bdd100k
        
        self.class_names = ['unlabelled','road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))
        # print(self.class_map)
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                    img_path.split(os.sep)[-2],
                                    os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')
        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)                           
        _target = Image.fromarray(_tmp)                                  

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':      
            return self.transform_val(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            # tr.ColorJitter(brightness=0.5, hue=0.3, contrast=0.2,saturation=0.2),
            # tr.RandomSizeAndCrop(size=self.args.crop_size, crop_nopad = False, ignore_index=255,pre_size=None),
            # tr.Resize(size1=self.args.crop_size, size2=self.args.crop_size),
            # tr.RandomGaussianBlur(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            #tr.FixScaleCrop(crop_size=self.args.val_crop_size),
            #tr.Resize(size1=self.args.val_sizeW, size2=self.args.val_sizeH),
            #tr.Contrast(),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)
    

class RainyCityscapesSegmentation(data.Dataset):
    NUM_CLASSES = 19  #19

    def __init__(self, args, root=Path.db_root_dir('rainy_cityscapes'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]    # for all datasets except bdd100k
        
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                    img_path.split(os.sep)[-2],
                                    os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')
        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)                           
        _target = Image.fromarray(_tmp)                                  

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':       #valid for GTAV else val
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            # tr.ColorJitter(brightness=0.5, hue=0.3, contrast=0.2,saturation=0.2),
            # tr.RandomSizeAndCrop(size=self.args.crop_size, crop_nopad = False, ignore_index=255,pre_size=None),
            # tr.Resize(size1=self.args.crop_size, size2=self.args.crop_size),
            # tr.RandomGaussianBlur(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            #tr.FixScaleCrop(crop_size=self.args.val_crop_size),
            #tr.Resize(size1=self.args.val_sizeW, size2=self.args.val_sizeH),
            #tr.Contrast(),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)




class Foggy_CityscapesSegmentation(data.Dataset):
    NUM_CLASSES = 19  #19

    def __init__(self, args, root=Path.db_root_dir('foggy_cityscapes'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit_foggy', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        if('beta_0.005' not in os.path.basename(img_path)):
            lbl_path = os.path.join(self.annotations_base,
                                    img_path.split(os.sep)[-2],
                                    os.path.basename(img_path)[:-31] + 'gtFine_labelIds.png')  #15
        else:
            lbl_path = os.path.join(self.annotations_base,
                                    img_path.split(os.sep)[-2],
                                    os.path.basename(img_path)[:-32] + 'gtFine_labelIds.png')
        
        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)                              
        _target = Image.fromarray(_tmp)                                   

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':       #valid for GTAV else val
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.ColorJitter(brightness=0.5, hue=0.3, contrast=0.2,saturation=0.2),
            tr.Resize(size1=self.args.base_size, size2=self.args.crop_size),
            tr.RandomGaussianBlur(),
            #tr.HPF(),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            #tr.Contrast(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            #tr.FixScaleCrop(crop_size=self.args.val_crop_size),
            #tr.Resize(size1=self.args.val_sizeW, size2=self.args.val_sizeH),
            #tr.Contrast(),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)
    
class GTAVSegmentation(data.Dataset):
    NUM_CLASSES = 19  #19

    def __init__(self, args, root=Path.db_root_dir('GTAV'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        self.images_base = os.path.join(self.root, 'images', self.split)
        self.annotations_base = os.path.join(self.root, 'labels', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34,-1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()    
        lbl_path = os.path.join(self.annotations_base,os.path.basename(img_path)[:-4] +'.png')
        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)                                    

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'test':       
            return self.transform_val(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.PASTA(3.0,0.25,2.0),
            tr.RandomHorizontalFlip(),
            tr.ColorJitter(brightness=0.5, hue=0.3, contrast=0.2,saturation=0.2),
            tr.RandomSizeAndCrop(size=self.args.crop_size, crop_nopad = False, ignore_index=255,pre_size=None),
            tr.Resize(size1=self.args.crop_size, size2=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.ToTensor()])
            #tr.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            # ])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            #tr.Resize(size1=1920, size2=1056),
            #tr.FixScaleCrop(crop_size=self.args.val_crop_size),
            #tr.Resize(size1=self.args.val_sizeW, size2=self.args.val_sizeH),
            #tr.Contrast(),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)
    
class BDD100kSegmentation(data.Dataset):
    NUM_CLASSES = 19  #19

    def __init__(self, args, root=Path.db_root_dir('BDD100k'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        self.images_base = os.path.join(self.root, 'images', self.split)
        self.annotations_base = os.path.join(self.root, 'labels', self.split)
        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.jpg')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
        
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,os.path.basename(img_path)[:-4] +'_train_id'+'.png')
        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':       #valid for GTAV else val
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.ColorJitter(brightness=0.5, hue=0.3, contrast=0.2,saturation=0.2),
            tr.Resize(size1=self.args.base_size, size2=self.args.crop_size),
            tr.RandomGaussianBlur(),
            #tr.HPF(),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            #tr.Contrast(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            #tr.Resize(size1=1280, size2=736),
            #tr.FixScaleCrop(crop_size=self.args.val_crop_size),
            #tr.Resize(size1=self.args.val_sizeW, size2=self.args.val_sizeH),
            #tr.Contrast(),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)
    
class SynthiaSegmentation(data.Dataset):
    NUM_CLASSES = 19  #19

    def __init__(self, args, root=Path.db_root_dir('SYNTHIA'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        self.images_base = os.path.join(self.root, 'RGB', self.split)
        self.annotations_base = os.path.join(self.root, 'GT/LABELS', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0,13,14,22]
        self.valid_classes = [3,4,2,21,5,7,15,9,6,16,1,10,17,8,18,19,20,12,11]
        
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,os.path.basename(img_path)[:-4] +'.png')
        _img = Image.open(img_path).convert('RGB')
        _tmp = np.asarray(imageio.imread(lbl_path, format='PNG-FI'))[:,:,0]
        label_copy = 255 * np.ones(_tmp.shape, dtype=np.float32)
        for k, v in self.class_map.items():
            label_copy[_tmp == k] = v
        _target = Image.fromarray(label_copy)                           
        
        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':       #valid for GTAV else val
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.ColorJitter(brightness=0.5, hue=0.3, contrast=0.2,saturation=0.2),
            tr.Resize(size1=self.args.base_size, size2=self.args.crop_size),
            tr.RandomGaussianBlur(),
            #tr.HPF(),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            #tr.Contrast(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            #tr.Resize(size1=1280, size2=768),
            #tr.FixScaleCrop(crop_size=self.args.val_crop_size),
            #tr.Resize(size1=self.args.val_sizeW, size2=self.args.val_sizeH),
            #tr.Contrast(),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


class MapillarySegmentation(data.Dataset):
    NUM_CLASSES = 19  #19

    def __init__(self, args, root=Path.db_root_dir('Mapillary'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        self.images_base = os.path.join(self.root, self.split, 'images')
        self.annotations_base = os.path.join(self.root, self.split, 'labels')

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.jpg')

        self.ignore_index = 255

        self.class_map = {}
        for i in range(66):
            self.class_map[i] = self.ignore_index

        ### Convert each class to cityscapes one
        ### Road
        # Road
        self.class_map[13] = 0
        # Lane Marking - General
        self.class_map[24] = 0
        # Manhole
        self.class_map[41] = 0

        ### Sidewalk
        # Curb
        self.class_map[2] = 1
        # Sidewalk
        self.class_map[15] = 1

        ### Building
        # Building
        self.class_map[17] = 2

        ### Wall
        # Wall
        self.class_map[6] = 3

        ### Fence
        # Fence
        self.class_map[3] = 4

        ### Pole
        # Pole
        self.class_map[45] = 5
        # Utility Pole
        self.class_map[47] = 5

        ### Traffic Light
        # Traffic Light
        self.class_map[48] = 6

        ### Traffic Sign
        # Traffic Sign
        self.class_map[50] = 7

        ### Vegetation
        # Vegitation
        self.class_map[30] = 8

        ### Terrain
        # Terrain
        self.class_map[29] = 9

        ### Sky
        # Sky
        self.class_map[27] = 10

        ### Person
        # Person
        self.class_map[19] = 11

        ### Rider
        # Bicyclist
        self.class_map[20] = 12
        # Motorcyclist
        self.class_map[21] = 12
        # Other Rider
        self.class_map[22] = 12

        ### Car
        # Car
        self.class_map[55] = 13

        ### Truck
        # Truck
        self.class_map[61] = 14

        ### Bus
        # Bus
        self.class_map[54] = 15

        ### Train
        # On Rails
        self.class_map[58] = 16

        ### Motorcycle
        # Motorcycle
        self.class_map[57] = 17

        ### Bicycle
        # Bicycle
        self.class_map[52] = 18

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,os.path.basename(img_path)[:-4] +'.png')
        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        mask = np.array(_tmp)
        mask_copy = mask.copy()
        for k, v in self.class_map.items():
            mask_copy[mask == k] = v
        _target = Image.fromarray(mask_copy.astype(np.uint8))                     
        
        sample = {'image': _img, 'label': _target}

        if self.split == 'training':
            return self.transform_tr(sample)
        elif self.split == 'validation':       
            return self.transform_val(sample)

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.ColorJitter(brightness=0.5, hue=0.3, contrast=0.2,saturation=0.2),
            tr.RandomCrop_p(self.args.base_size, self.args.crop_size),
            tr.RandomGaussianBlur(),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            #tr.Resize(size1=self.args.val_sizeW, size2=self.args.val_sizeH),
            tr.ResizeHeight(self.args.eval_size),
            tr.CenterCropPad(self.args.eval_size),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


parser = argparse.ArgumentParser()
args = parser.parse_args()
args.base_size = 768
args.crop_size = 768
args.eval_size = 1536
args.val_sizeH = 512 
args.val_sizeW = 1024 
args.model_name = 'cvpr_bsz16mrfpplusrun3'

cityscapes_train = CityscapesSegmentation(args, split='train')
cityscapes_val = CityscapesSegmentation(args, split='val')

GTAV_train = GTAVSegmentation(args, split='train')
GTAV_val = GTAVSegmentation(args, split='test')

BDD_train = BDD100kSegmentation(args, split='train')
BDD_val = BDD100kSegmentation(args, split='val')

synthia_train = SynthiaSegmentation(args, split='train')
synthia_val = SynthiaSegmentation(args, split='val')

mapillary_train = MapillarySegmentation(args, split='training')
mapillary_val = MapillarySegmentation(args, split='validation')

city_train_dataloader = DataLoader(cityscapes_train, batch_size=16, shuffle=True, num_workers=4,pin_memory=True)
train_dataloader = DataLoader(GTAV_train, batch_size=16, shuffle=True, num_workers=8,pin_memory=True) #batch_size=16
synthia_train_dataloader = DataLoader(synthia_train, batch_size=8, shuffle=True, num_workers=4,pin_memory=True)
city_val_dataloader = DataLoader(cityscapes_val, batch_size=1, shuffle=False,pin_memory=True, num_workers=4)
GTAV_val_dataloader = DataLoader(GTAV_val, batch_size=1, shuffle=False,pin_memory=True, num_workers=4)
BDD_val_dataloader = DataLoader(BDD_val, batch_size=1, shuffle=False,pin_memory=True, num_workers=4)
synthia_val_dataloader = DataLoader(synthia_val, batch_size=1, shuffle=False,pin_memory=True, num_workers=4)
mapillary_val_loader = DataLoader(mapillary_val, batch_size=1, shuffle=False,pin_memory=True, num_workers=4)
GTA_Synthia_trainloader = ConcatDataset([GTAV_train,synthia_train])
criterion = nn.CrossEntropyLoss(ignore_index=255)  #Unet_ibnnet  Unet_MRFP

model = nn.DataParallel(LabMAO_MAOEncPerturb_noskipconn_randomOC_godmodel_INAffineTrue_perturbindec(num_classes=19, criterion=criterion).to('cuda:0'),device_ids=[0])

#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2,momentum=0.9,weight_decay=5e-4)
max_loss = 10000.0
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))


class LRPolicy(object):
    def __init__(self, powr):
        self.powr = powr

    def __call__(self, iter):
        return math.pow(1-iter/40000, self.powr)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LRPolicy(powr=0.9))

MODEL_PATH = '/home/user/Perception/SDG/ckpt/Robustnet_settings_all_baseline_107epochs'
iter = 0
ended = 0
for epoch in range(110):
    iterator = tqdm(train_dataloader)
    model.train()
    torch.cuda.empty_cache()
    izz=0
    for sample in iterator:
        izz +=1
        iter +=1
        if iter>39998:
            ended=1       
            torch.cuda.empty_cache()
            break
        img = sample['image'].cuda(non_blocking=True).to('cuda:0')
        label = sample['label'].cuda(non_blocking=True).to('cuda:0')

        loss = model(img,label.long(),training=True)
        optimizer.zero_grad()        
        loss.backward()        
        optimizer.step()
        scheduler.step()

        iterator.set_description("epoch: {}; iter: {}; l_r: {:.7f}; Loss {:.4f} ".format(epoch, iter, scheduler.get_lr()[-1], loss))
        if iter>39980:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(MODEL_PATH, args.model_name+'_actuallatest.pth'))
    torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, os.path.join(MODEL_PATH, args.model_name+'_latest.pth'))
    if ended==1:
        break


print("Done training! ")
print("------------------------------------Validation after 40k iterations-----------------------------")
test_domains = [BDD_val_dataloader, city_val_dataloader, synthia_val_dataloader, mapillary_val_loader, GTAV_val_dataloader]
test_domains_str = ['BDD100k', 'Cityscapes', 'SYNTHIA','Mapillary', 'GTAV']

for i in range(len(test_domains)):
    test_iterator = tqdm(test_domains[i])
    checkpoint = torch.load(os.path.join(MODEL_PATH,args.model_name+'_actuallatest.pth'))

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    mIoU = []
    iou_acc = 0
    iy = 0
    for sample in test_iterator:
        img = sample['image'].cuda(non_blocking=True).to('cuda:0')
        label = sample['label'].cuda(non_blocking=True).to('cuda:0')
        if(img.shape[2:]==label.shape[1:]):
            
            outputs = model(img,training=False)
            
            pred = outputs.data.cpu().numpy()
            predictions = outputs.data.cpu().numpy()
            predictions = predictions[0]

            predictions = np.argmax(predictions, axis=0)

            target = label.cpu().numpy().astype('int64')
            
            pred = np.argmax(pred, axis=1)

            iou_acc += metrics.fast_hist(pred.flatten(), target.flatten(),
                                    19)
        else:
            iy+=1
    print("Number of images dropped:",iy)
    metrics.evaluate_eval(iou_acc, dataset_name=test_domains_str[i])

