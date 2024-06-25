import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2
from PIL import Image
from torchvision import transforms
import torch
from PIL import Image, ImageEnhance
import scipy.ndimage
import scipy.fft
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders.unet import UnetDecoder
from typing import Optional, Union, List

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
from utils import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import timeit
import metrics
from torchmetrics import CosineSimilarity
import random
import warnings
import copy
img_path1 = '/home/user/Perception/SDG/spectrum_analysis_imgs/1.png'

"""img_path1 = './frankfurt_000001_004736_leftImg8bit.png'
#img_path2 = '/home/user/Perception/SDG/2.png'
sup_res_img = cv2.imread(img_path1)
sup_res_img=cv2.cvtColor(sup_res_img, cv2.COLOR_BGR2GRAY)
sup_res_img = sup_res_img[0:1024,0:1024]
########### HPF and LPF ####################################
dft = cv2.dft(np.float32(sup_res_img), flags=cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
rows, cols = sup_res_img.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask_HPF = np.ones((rows, cols, 2), np.uint8)
mask_LPF = np.ones((rows, cols, 2), np.uint8)
r = 69
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area_HPF = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask_area_LPF = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= r*r

mask_LPF[mask_area_LPF] = 0
mask_HPF[mask_area_HPF] = 0
fshift_mask_mag_HPF = dft_shift * mask_HPF
fshift_mask_mag_LPF = dft_shift * mask_LPF
f_ishift_HPF = np.fft.ifftshift(fshift_mask_mag_HPF)
img_back_HPF = cv2.idft(f_ishift_HPF)
img_back_HPF = cv2.magnitude(img_back_HPF[:, :, 0], img_back_HPF[:, :, 1])


plt.imshow(img_back_HPF)
plt.savefig('./HPF_frankfurt.png')
print(hey)
fft_HPF = np.fft.fftn(img_back_HPF)
fft_amp_HPF = np.abs(fft_HPF)
phase_img_HPF = np.fft.ifftn((fft_HPF/fft_amp_HPF))
phase_img_HPF = (phase_img_HPF*5*255).astype(np.uint8)

f_ishift_LPF = np.fft.ifftshift(fshift_mask_mag_LPF)
img_back_LPF = cv2.idft(f_ishift_LPF)
img_back_LPF = cv2.magnitude(img_back_LPF[:, :, 0], img_back_LPF[:, :, 1])

fft_LPF = np.fft.fftn(img_back_LPF)
fft_amp_LPF = np.abs(fft_LPF)
phase_img_LPF = np.fft.ifftn((fft_LPF/fft_amp_LPF))
phase_img_LPF = (phase_img_LPF*5*255).astype(np.uint8)
###########################################################

fft_res_img = np.fft.fftn(sup_res_img)
fft_amp = np.abs(fft_res_img)
phase_img = np.fft.ifftn((fft_res_img/fft_amp))
phase_img = (phase_img*5*255).astype(np.uint8)

median_phase_img = scipy.ndimage.median_filter(phase_img,size=2)
plt.imshow(median_phase_img)
plt.savefig('./median_phase_img.jpg')
plt.close()

fig = plt.figure(figsize=(9, 9))

ax1 = fig.add_subplot(3,3,1)
ax1.imshow(sup_res_img, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(3,3,2)
ax2.imshow(magnitude_spectrum, cmap='gray')
ax2.title.set_text('FFT of image')
ax3 = fig.add_subplot(3,3,3)
ax3.imshow(img_back_HPF, cmap='gray')
ax3.title.set_text('High pass filter')
ax3 = fig.add_subplot(3,3,4)
ax3.imshow(img_back_LPF, cmap='gray')
ax3.title.set_text('Low pass filter')
ax4 = fig.add_subplot(3,3,5)
ax4.imshow(phase_img, cmap='gray')
ax4.title.set_text('PHOT')
ax5 = fig.add_subplot(3,3,6)
ax5.imshow(phase_img_HPF, cmap='gray')
ax5.title.set_text('PHOT_HPF')
ax6 = fig.add_subplot(3,3,7)
ax6.imshow(phase_img_LPF, cmap='gray')
ax6.title.set_text('PHOT_LPF')
plt.savefig('Fourier_1.jpg')
plt.close()
plt.imshow(img_back_LPF)
plt.savefig('./LPF24.jpg')
plt.close()
plt.imshow(phase_img)
plt.savefig('./PHOT.jpg')
plt.close()"""

#img1 = skimage.io.imread(img_path1)
#img1 = cv2.imread(img_path1)
#img1 =  cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#gauss_noise=np.zeros((457,512),dtype=np.uint8)
#gauss_noise = cv2.randn(gauss_noise,128,20)
#gauss_noise=(gauss_noise*0.5).astype(np.uint8)
#img1=cv2.add(img1,gauss_noise)
#img2 = skimage.io.imread(img_path2)
#img1 = Image.open(img_path1)
#img = Image.fromarray(img1.astype('uint8'), 'RGB')
#img = ImageEnhance.Contrast(img).enhance(1)
#img = np.array(img)
#img = cv2.imread(img_path)
#img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(img1)
#plt.savefig('./spectrum_analysis_imgs/noisy_lena.png')
#plt.close()

'''fft_img1=np.fft.fftn(img1)
fft_amplitude1 = np.abs(fft_img1)
fft_phase1 =  np.angle(fft_img1)
phase_img = fft_img1/fft_amplitude1
#fft_amplitude2, fft_phase2 = np.abs(fft_img2), np.angle(fft_img2)

#fft_new_real, fft_new_imag = np.real(fft_img1)/fft_amplitude1, np.imag(fft_img1)/fft_amplitude1
#fft_new_amplitude1, fft_new_phase1 = np.sqrt(np.square(fft_new_real)+np.square(fft_new_imag)), np.arctan2(fft_new_imag,fft_new_real)

#phase_img = np.exp((1j) * fft_phase1)
phase_img = np.fft.ifftn(fft_amplitude1)

phase_img = (phase_img).astype(np.uint8)
print(np.unique(phase_img))
phase_img = Image.fromarray(phase_img)
plt.imshow(phase_img)
#skimage.io.imsave('lena_phase.png')
plt.savefig('./spectrum_analysis_imgs/lena_amp_img.png')
print(hey)'''
'''amp_img = fft_amplitude*np.exp((1j)*0)
amp_img = np.fft.ifftn(amp_img)
plt.imshow(np.real(amp_img))
plt.savefig('./amp_img.png')'''

##APR##

'''one_percent_phase_recon = fft_amplitude2*np.exp((1j)*(fft_phase1))
perturbed_phase_recon = np.fft.ifftn(np.fft.ifftshift(one_percent_phase_recon))
perturbed_phase_recon = perturbed_phase_recon.astype(np.uint8)
perturbed_phase_recon = Image.fromarray(perturbed_phase_recon)
plt.imshow(perturbed_phase_recon)
plt.savefig('./amp2phase1.png')

one_percent_amp_recon = (fft_amplitude1)*np.exp((1j) * fft_phase2)
perturbed_amp_recon = np.fft.ifftn(np.fft.ifftshift(one_percent_amp_recon))
perturbed_amp_recon = perturbed_amp_recon.astype(np.uint8)
perturbed_amp_recon = Image.fromarray(perturbed_amp_recon)
plt.imshow(perturbed_amp_recon)
plt.savefig('./amp1phase2.png')
print("done!")'''

## 1 percent stuff ## 

'''one_percent_phase_recon = fft_amplitude1*np.exp((1j)*(fft_phase1*0.1961))
perturbed_phase_recon = np.fft.ifftn(np.fft.ifftshift(one_percent_phase_recon))
perturbed_phase_recon = perturbed_phase_recon.astype(np.uint8)
perturbed_phase_recon = Image.fromarray(perturbed_phase_recon)
plt.imshow(perturbed_phase_recon)
plt.savefig('./point1961*phase_phase.png')

one_percent_amp_recon = (fft_amplitude1*3)*np.exp((1j) * fft_phase1)
perturbed_amp_recon = np.fft.ifftn(np.fft.ifftshift(one_percent_amp_recon))
perturbed_amp_recon = perturbed_amp_recon.astype(np.uint8)
perturbed_amp_recon = Image.fromarray(perturbed_amp_recon)
plt.imshow(perturbed_amp_recon)
plt.savefig('./3*amp_amp.png')'''


'''img = skimage.io.imread(img_path1)
img = torch.transpose(torch.from_numpy(img),2,0)
img = torch.transpose(img,1,2)

#functions
def create_mask(ths, ref_fft):
    #null mask
    mask = torch.ones((ref_fft.shape), dtype=torch.float32)
    _,_, h, w = ref_fft.shape
    b_h = np.floor((h*ths)/2.0).astype(int)
    b_w = np.floor((w*ths)/2.0).astype(int)
    if b_h == 0 and b_w ==0:
        return mask
    else:
        mask[:,:, 0:b_h, 0:b_w]     = 0      # top left
        mask[:,:, 0:b_h, w-b_w:w]   = 0      # top right
        mask[:,:, h-b_h:h, 0:b_w]   = 0      # bottom left
        mask[:,:, h-b_h:h, w-b_w:w] = 0      # bottom right

    return mask

n = 32
ths = [0.0, 1.0]

ths=list(np.arange(ths[0], ths[1]+1/n, 1/n))

fft_source = torch.zeros([1, 3, 1024, 2048])
fft_source_masks = []
for i in range(len(ths)-1):
    t1 = ths[i]
    t2 = ths[i+1]
    mask = create_mask(t1, fft_source) - create_mask(t2, fft_source) #(1, 3, h, w, 2)
    mask = torch.unsqueeze(mask, 1)
    fft_source_masks.append(mask)
fc_fft_source_masks = torch.cat(fft_source_masks, dim=1) #(1, n, 3, h, w, 2)
#print(fc_fft_source_masks.shape)
###########################################################################################
n = 32
#transform
#img = cv2.imread(img_path1)

#img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fft_input = torch.fft.fftn(img.clone()).unsqueeze(0)

b, c, im_h, im_w = fft_input.shape

# extract amplitude and phase of both ffts (1, 3, h, w)
amp_src, pha_src = torch.abs(fft_input.clone()), torch.angle(fft_input.clone())

#band_pass filter
amp_src_32 = torch.unsqueeze(amp_src, 1) #(1, 1, 3, h, ...)
#print(amp_src_32.shape)

amp_src_32 = amp_src_32.expand((b, n, c, im_h, im_w)) #(1, 32, 3, h, ...)

amp_src_32 = amp_src_32 * torch.real(fc_fft_source_masks[:, :, :, :, :]) #(1, n, 3, h, w)
amp_src_96 = amp_src_32.reshape(b,c*n, im_h, im_w)

amp_src_96_faa = torch.zeros((amp_src_96.shape), dtype=torch.float32)
amp_src_96_faa[:,6:48,:,:] = 0#amp_src_96[:,6:48,:,:]
amp_src_96_faa[:,0:6,:,:] = 0
amp_src_96_faa[:,48:96,:,:] = amp_src_96[:,48:96,:,:]#0

amp_src_32_faa = amp_src_96_faa.view(b, n, c, im_h, im_w)
amp_src_ = torch.sum(amp_src_32_faa, dim=1)


real = torch.cos(pha_src.clone()) * amp_src_.clone()
imag = torch.sin(pha_src.clone()) * amp_src_.clone()
fft_input_ = torch.complex(real,imag)

image_bandpass = torch.fft.ifftn(fft_input_).squeeze(0)
image_bandpass = torch.transpose(image_bandpass,2,0)
print(image_bandpass.shape)
image_bandpass = torch.transpose(image_bandpass,0,1)
print(image_bandpass.shape)
image_bandpass = image_bandpass.numpy()
image_bandpass = np.real(image_bandpass).astype(np.uint8)
plt.imshow(image_bandpass)
plt.savefig('./highpass_filter.png')

print("done!")'''

class Unet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "efficientnet-b3",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = False,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 19
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        
        self.decoder = UnetDecoder(
            encoder_channels=(3,40,32,48,136,384),
  
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,

        )


        
    def forward(self, x, training):

        features = self.encoder(x)

        decoder_output = self.decoder(*features)
        logit = self.segmentation_head(decoder_output)
        return logit, features
    



class CityscapesSegmentation(data.Dataset):
    NUM_CLASSES = 19  #19

    def __init__(self, args, root=Path.db_root_dir('cityscapes'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        #self.valid_classes = [11]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']
        #self.class_names = ['building']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        #print(img_path)
        '''if('beta_0.005' not in os.path.basename(img_path)):
            lbl_path = os.path.join(self.annotations_base,
                                    img_path.split(os.sep)[-2],
                                    os.path.basename(img_path)[:-31] + 'gtFine_labelIds.png')  #15
        else:
            lbl_path = os.path.join(self.annotations_base,
                                    img_path.split(os.sep)[-2],
                                    os.path.basename(img_path)[:-32] + 'gtFine_labelIds.png')'''
        lbl_path = os.path.join(self.annotations_base,
                                    img_path.split(os.sep)[-2],
                                    os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        _img = Image.open(img_path).convert('RGB')
        #print(np.unique(skimage.io.imread(lbl_path)))
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        #print(np.unique(_tmp))
        #print(hey)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
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
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            #tr.HPF(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            #tr.Contrast(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            #tr.Contrast(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.base_size = 512
args.crop_size = 512

cityscapes_train = CityscapesSegmentation(args, split='train')
cityscapes_val = CityscapesSegmentation(args, split='val')
#cityscapes_test = CityscapesSegmentation(args, split='test')

train_dataloader = DataLoader(cityscapes_train, batch_size=8, shuffle=True, num_workers=4)
val_dataloader = DataLoader(cityscapes_val, batch_size=1, shuffle=False)

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None
revgrad = GradientReversal.apply

class GradientReversal_Layer(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)


class DeepMAO(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "efficientnet-b3",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = False,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 19
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        
        self.decoder = UnetDecoder(
            encoder_channels=(3,40,32,48,136,384),
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
        )

        self.OClayer1 = nn.Conv2d(40,56,kernel_size=3, stride=1, padding=1)
        self.OC1_bn = nn.BatchNorm2d(56)
        self.OClayer2 = nn.Conv2d(56,64,kernel_size=3, stride=1, padding=1)
        self.OC2_bn = nn.BatchNorm2d(64)
        self.OClayer3 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1)
        self.OC3_bn = nn.BatchNorm2d(128)
        self.OClayer4 = nn.Conv2d(128,16,kernel_size=3, stride=1, padding=1)
        self.OC4_bn = nn.BatchNorm2d(16)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hv = nn.Conv2d(16,16,kernel_size=1,stride=1,padding=1)
        self.ht = nn.Conv2d(16,16,kernel_size=1,stride=1,padding=1)
        self.z = nn.Conv2d(32,16,kernel_size=1,stride=1,padding=1)
        self.GRL = GradientReversal_Layer(alpha=1.0)

        #self.randconv1 = nn.Conv2d(56,32,kernel_size=1, stride=1, padding=1).requires_grad_(False)
        #self.randconv2 = nn.Conv2d(64,48,kernel_size=1, stride=1, padding=1).requires_grad_(False)

    '''def Normalization_Perturbation(self, feat):
    # feat: input features of size (B, C, H, W)
        feat_mean = feat.mean((2, 3), keepdim=True) # size: B, C, 1, 1
        ones_mat = torch.ones_like(feat_mean)
        alpha = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
        beta = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
        output = alpha * feat - alpha * feat_mean + beta * feat_mean
        return output # size: B, C, H, W'''

 
    def forward(self, x,training):
        '''features = []
        features.append(x)
        x = self.encoder._conv_stem(x)
        x = self.encoder._bn0(x)  ## Input to NP
        features.append(x)
        p = random.random()
        OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(features[1]),scale_factor =(1.2,1.2)))) #layersize256 #output320
        if(training==True and p<0.5):
            OCout = self.Normalization_Perturbation(OCout)
            self.randconv1.weight = nn.init.kaiming_normal_(self.randconv1.weight,mode='fan_out', nonlinearity='relu')
            rand_OCout = F.interpolate(self.randconv1(OCout), size=(128,128))
            rand_OCout = self.GRL(rand_OCout)
        else:
            rand_OCout = torch.tensor([0.0]).to('cuda:0')

        for i in range(5):
            x = self.encoder._blocks[i](x)
        x = x+rand_OCout
        features.append(x)
        
        for i in range(5,8):
            x = self.encoder._blocks[i]((x))

        OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))#layersize320 #output400
        if(training==True and p<0.5):
            OCout = self.Normalization_Perturbation(OCout)
            self.randconv2.weight = nn.init.kaiming_normal_(self.randconv2.weight,mode='fan_out', nonlinearity='relu')
            rand_OCout = F.interpolate(self.randconv2(OCout), size=(64,64))
            rand_OCout = self.GRL(rand_OCout)
        else:
            rand_OCout = torch.tensor([0.0]).to('cuda:0')

        x = x+rand_OCout
        features.append(x)
        
        for i in range(8,18):
            x = self.encoder._blocks[i](x)
        features.append(x)

        for i in range(18,26):
            x = self.encoder._blocks[i](x)
        features.append(x)'''
        
        features = self.encoder(x)

        OCout1 = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(features[1]),scale_factor =(1.2,1.2)))) #layersize256 #output320
        OCout2 = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout1), scale_factor =(1.2,1.2))))#layersize320 #output400
        OCout3 = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout2), scale_factor =(1.2,1.2))))#layersize400 output500
        OCout = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout3), scale_factor =(1.15,1.15))))#layersize500 output625
        logit_dec = self.decoder(*features) #16
        _,_,h,w = logit_dec.shape

        if(logit_dec.shape==OCout.shape):
            #logit = torch.add(OCout, logit)
            hv = self.tanh(self.hv(logit_dec))
            ht = self.tanh(self.ht(OCout))
            z = self.sigmoid(self.z(torch.cat([logit_dec,OCout],dim=1)))
            logit = z*hv + (1-z)*ht
        else:
            OCout = F.interpolate(OCout,size=(h,w),mode='bilinear')
            #logit = torch.add(OCout, logit)
            hv = self.tanh(self.hv(logit_dec))
            ht = self.tanh(self.ht(OCout))
            z = self.sigmoid(self.z(torch.cat([logit_dec,OCout],dim=1)))
            logit = F.interpolate((z*hv + (1-z)*ht),size=(h,w),mode='bilinear')

        logit = self.segmentation_head(logit)
        
        return logit, logit_dec, OCout,features, OCout1,OCout2,OCout3
    
model = DeepMAO().to('cuda:0')
MODEL_PATH = '/home/user/Perception/SDG/ckpt/Unet_n_Deepmao_Cityscapes/deepmao_bsz8_GMU_fusion.pth'

#MODEL_PATH = '/home/user/Perception/SDG/ckpt/Unet_n_Deepmao_Cityscapes/cityscapes_unet_bsz8.pth'

'''#a = torch.Tensor([10,11,10,11,10,11,0])
#b = torch.Tensor([10.5,11,10.5,11,10.5,11,10.5])
a = torch.Tensor([0,0,0,0,0,0,0,0,5])
plt.hist(a)
plt.savefig('./a.png')
#print(torch.mean(a))
#print(torch.median(a))
#b = torch.Tensor([0,1,0,1,0,1,0,1,0])
#b = torch.Tensor([10.5,11,10.5,11,10.5,11,10.5])
#fft_a = torch.fft.fftn(a)
#print(fft_a)
#fft_b = torch.fft.fftn(b)
#print(fft_b)
#fft_a_mean = torch.mean(torch.abs(fft_a))
#fft_b_mean = torch.mean(torch.abs(fft_b))
#print(fft_a_mean,fft_b_mean)
print(hey)'''

test_iterator = tqdm(val_dataloader)
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
mIoU = []
i=0
for sample in test_iterator:
    img = sample['image'].cuda(non_blocking=True).to('cuda:0')
    label = sample['label'].cuda(non_blocking=True).to('cuda:0')
    outputs, feat, OCout,features, OCout1,OCout2,OCout3 = model(img, False)
    #outputs,features = model(img, False)
    i +=1
    if(i<=3):
        #fft_feat0 = torch.fft.fftn(features[0], dim=(2,3))
        fft_feat1 = torch.fft.fftn(OCout1, dim=(2,3))

        fft_shift_img = torch.fft.fftshift(fft_feat1)
        
        bsz,ch,_,_ = fft_shift_img.shape
        rows, cols = fft_feat1.shape[2:]
        crow, ccol = int(rows / 2), int(cols / 2)

        mask_HPF = np.ones((bsz, ch, rows, cols), np.uint8)
        
        r = 100
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area_HPF = (x - center[0]) ** 2 + (y - center[1]) ** 2<= r*r

        mask_HPF[:,:,mask_area_HPF] = 0
        fft_high_img = fft_shift_img * (torch.from_numpy(mask_HPF).to('cuda:0'))
        print("HPF values in feat1:", torch.count_nonzero(torch.abs(fft_shift_img-fft_high_img)))
        print("HPF values mean in feat1:", torch.max(torch.abs(fft_shift_img-fft_high_img)))
        
        x_img = torch.abs(fft_shift_img-fft_high_img).detach().cpu()
        #print(x_img.shape)
        x_img_channel = x_img[0][0].numpy()
        print(np.max(x_img_channel))
        print(np.min(x_img_channel))

        plt.hist(x_img_channel, bins=np.arange(np.min(x_img_channel), 10, step=1))
        plt.ylim([0, 400])
        plt.savefig('./aaaaa_OCout.png')
        print(hey)
        #print(hey)
        # inverce fft
        ifft_img1 = torch.fft.ifftn(fft_high_img)
        #print(ifft_img1)


        fft_OCout1 = torch.fft.fftn(OCout1, dim=(2,3))

        fft_shift_img = torch.fft.fftshift(fft_OCout1)
        
        bsz,ch,_,_ = fft_shift_img.shape
        rows, cols = fft_OCout1.shape[2:]
        crow, ccol = int(rows / 2), int(cols / 2)

        mask_HPF = np.ones((bsz, ch, rows, cols), np.uint8)
        
        r = 100
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area_HPF = (x - center[0]) ** 2 + (y - center[1]) ** 2<= r*r

        mask_HPF[:,:,mask_area_HPF] = 0
        fft_high_img = fft_shift_img * (torch.from_numpy(mask_HPF).to('cuda:0'))
        print("HPF values in OCout1", torch.count_nonzero(torch.abs(fft_shift_img-fft_high_img)))
        print("HPF values mean in OCout1:", torch.max(torch.abs(fft_shift_img-fft_high_img)))
        print(hey)
        # inverce fft
        ifft_img1 = torch.fft.ifftn(fft_high_img)
        #print(ifft_img1)


        fft_feat2 = torch.fft.fftn(features[2], dim=(2,3))
        fft_shift_img = torch.fft.fftshift(fft_feat2)
        filter_rate = 0.5
        h, w = fft_shift_img.shape[:2] # height and width
        cy, cx = int(h/2), int(w/2) # centerness
        rh, rw = int(filter_rate * cy), int(filter_rate * cx) # filter_size
        # the value of center pixel is zero.
        fft_shift_img[cy-rh:cy+rh, cx-rw:cx+rw] = 0
        # restore the frequency image
        ifft_shift_img2 = torch.fft.ifftshift(fft_shift_img)


        fft_feat3 = torch.fft.fftn(features[3], dim=(2,3))
        fft_shift_img = torch.fft.fftshift(fft_feat3)
        filter_rate = 0.5
        h, w = fft_shift_img.shape[:2] # height and width
        cy, cx = int(h/2), int(w/2) # centerness
        rh, rw = int(filter_rate * cy), int(filter_rate * cx) # filter_size
        # the value of center pixel is zero.
        fft_shift_img[cy-rh:cy+rh, cx-rw:cx+rw] = 0
        # restore the frequency image
        ifft_shift_img3 = torch.fft.ifftshift(fft_shift_img)


        fft_feat4 = torch.fft.fftn(features[4], dim=(2,3))
        fft_shift_img = torch.fft.fftshift(fft_feat4)
        filter_rate = 0.5
        h, w = fft_shift_img.shape[:2] # height and width
        cy, cx = int(h/2), int(w/2) # centerness
        rh, rw = int(filter_rate * cy), int(filter_rate * cx) # filter_size
        # the value of center pixel is zero.
        fft_shift_img[cy-rh:cy+rh, cx-rw:cx+rw] = 0
        # restore the frequency image
        ifft_shift_img4 = torch.fft.ifftshift(fft_shift_img)


        fft_feat5 = torch.fft.fftn(features[5], dim=(2,3))
        fft_shift_img = torch.fft.fftshift(fft_feat5)
        filter_rate = 0.5
        h, w = fft_shift_img.shape[:2] # height and width
        cy, cx = int(h/2), int(w/2) # centerness
        rh, rw = int(filter_rate * cy), int(filter_rate * cx) # filter_size
        # the value of center pixel is zero.
        fft_shift_img[cy-rh:cy+rh, cx-rw:cx+rw] = 0
        # restore the frequency image
        ifft_shift_img5 = torch.fft.ifftshift(fft_shift_img)

        #fft_feat0_mean = torch.mean(torch.abs(fft_feat0))
        fft_feat1_mean = torch.mean(torch.abs(ifft_shift_img1))
        fft_feat2_mean = torch.mean(torch.abs(ifft_shift_img2))
        fft_feat3_mean = torch.mean(torch.abs(ifft_shift_img3))
        fft_feat4_mean = torch.mean(torch.abs(ifft_shift_img4))
        fft_feat5_mean = torch.mean(torch.abs(ifft_shift_img5))

        '''fft_OCout1 = torch.fft.fftn(OCout1, dim=(2,3))
        fft_OCout2 = torch.fft.fftn(OCout2, dim=(2,3))
        fft_OCout3 = torch.fft.fftn(OCout3, dim=(2,3))
        fft_OCout4 = torch.fft.fftn(OCout, dim=(2,3))

        fft_feat1_mean = torch.mean(torch.abs(fft_OCout1))
        fft_feat2_mean = torch.mean(torch.abs(fft_OCout2))
        fft_feat3_mean = torch.mean(torch.abs(fft_OCout3))
        fft_feat4_mean = torch.mean(torch.abs(fft_OCout4))'''

        print(fft_feat1_mean, fft_feat2_mean,fft_feat3_mean,fft_feat4_mean, fft_feat5_mean)
        print('\n')
    else:
        break
    #pred = outputs.data.cpu().numpy()
    #target = label.cpu().numpy()
    #pred = np.argmax(pred, axis=1)
    #iou = metrics.eval(target,pred)
    #mIoU.append(iou)

#print("mIoU:{}".format(np.sum(mIoU)/len(val_dataloader)))