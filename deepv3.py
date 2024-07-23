"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
import math
from re import L
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from network import Resnet
from network.mynn import initialize_weights, Norm2d, Upsample, initialize_weights_kaimingnormal_forOC
import numpy as np
import random
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward, DWTInverse
from torch.backends import cudnn
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
from typing import Optional, Union, List


from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders.unet import UnetDecoder

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import modules as md

# cudnn.deterministic = True
# cudnn.benchmark = False


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn
        print("output_stride = ", output_stride)
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            Norm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()
        
        img_features = self.img_pooling(x)
        
        img_features = self.img_conv(img_features)
        
        img_features = Upsample(img_features, x_size[2:])
        out = img_features
        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

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

        

class DeepV3Plus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,0,0,0,0,0], use_wtloss=False):
        super(DeepV3Plus, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.variant = variant
        self.wt_layer = wt_layer
        self.use_wtloss = use_wtloss
        self.trunk = trunk
        
        channel_3rd = 256
        prev_final_channel = 1024
        final_channel = 2048
        
        if trunk == 'resnet-50':
            resnet = Resnet.resnet50(wt_layer=self.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            # raise 'unknown deepv3 variant: {}'.format(self.variant)
            print("Not using Dilation ")

        os=16  ######### D16 ###############

        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                    output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            Norm2d(64),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1, bias=True))

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        self.three_input_layer = False

    def Normalization_Perturbation(self, feat):
    # feat: input features of size (B, C, H, W)
        feat_mean = feat.mean((2, 3), keepdim=True) # size: B, C, 1, 1
        ones_mat = torch.ones_like(feat_mean)
        alpha = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
        beta = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
        output = alpha * feat - alpha * feat_mean + beta * feat_mean
        return output # size: B, C, H, W
    
    def Normalization_Perturbation_Plus(self, feat):
        feat_mean = feat.mean((2, 3), keepdim=True)
        ones_mat = torch.ones_like(feat_mean)
        zeros_mat = torch.zeros_like(feat_mean)
        mean_diff = torch.std(feat_mean, 0, keepdim=True)
        mean_scale = mean_diff / mean_diff.max() * 1.5
        alpha = torch.normal(ones_mat, 0.75 * ones_mat)
        beta = 1 + torch.normal(zeros_mat, 0.75 * ones_mat) * mean_scale
        output = alpha * feat - alpha * feat_mean + beta * feat_mean
        return output

    def forward(self, x, gts=None,training=False):
        p = random.random()
        w_arr = []
        x_hfl = []
        x_size = x.size()  # 800
        # ResNet
        x = self.layer0[0](x)

        x = self.layer0[1](x)

        x = self.layer0[2](x)

        x = self.layer0[3](x)

        x_tuple = self.layer1([x, w_arr])  # 400

        low_level = x_tuple[0]

        x_tuple = self.layer2(x_tuple)  # 100
        
        x_tuple = self.layer3(x_tuple)  # 100

        x_tuple = self.layer4(x_tuple)  # 100
        
        x = x_tuple[0]
        w_arr = x_tuple[1]

        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)


        dec0_fine = self.bot_fine(low_level)

        dec0_up = Upsample(dec0_up, low_level.size()[2:])

        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)

        dec1 = self.final1(dec0)

        dec2 = self.final2(dec1)


        main_out = Upsample(dec2, x_size[2:])

        if training:
            loss1 = self.criterion(main_out, gts)
            return_loss = loss1
            return return_loss
        else:
            return main_out

        
class LabMAO_MAOEncPerturb_noskipconn_randomOC_godmodel(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,0,0,0,0,0], use_wtloss=False):
        super(LabMAO_MAOEncPerturb_noskipconn_randomOC_godmodel, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.variant = variant
        self.wt_layer = wt_layer
        self.use_wtloss = use_wtloss
        self.trunk = trunk
        
        channel_3rd = 256
        prev_final_channel = 1024
        final_channel = 2048
        
        if trunk == 'resnet-50':
            resnet = Resnet.resnet50(wt_layer=self.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            # raise 'unknown deepv3 variant: {}'.format(self.variant)
            print("Not using Dilation ")

        os=16  ######### D16 ###############

        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                    output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))
        ########################################### Stage 1 ##################################################################
        self.OClayer1 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC1_bn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OClayer2 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC2_bn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OClayer3 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC3_bn = nn.BatchNorm2d(128).requires_grad_(False)
        self.OClayer4 = nn.Conv2d(128,256,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC4_bn = nn.BatchNorm2d(256).requires_grad_(False)

        self.OCdeclayer1 = nn.Conv2d(256,128,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC1_decbn = nn.BatchNorm2d(128).requires_grad_(False)
        self.OCdeclayer2 = nn.Conv2d(128,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC2_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC3_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer4 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC4_decbn = nn.BatchNorm2d(64).requires_grad_(False)

        initialize_weights_kaimingnormal_forOC(self.OClayer1)
        initialize_weights_kaimingnormal_forOC(self.OC1_bn)
        initialize_weights_kaimingnormal_forOC(self.OClayer2)
        initialize_weights_kaimingnormal_forOC(self.OC2_bn)
        initialize_weights_kaimingnormal_forOC(self.OClayer3)
        initialize_weights_kaimingnormal_forOC(self.OC3_bn)
        initialize_weights_kaimingnormal_forOC(self.OClayer4)
        initialize_weights_kaimingnormal_forOC(self.OC4_bn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
        initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
        initialize_weights_kaimingnormal_forOC(self.OC2_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer3)
        initialize_weights_kaimingnormal_forOC(self.OC3_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer4)
        initialize_weights_kaimingnormal_forOC(self.OC4_decbn)

        # ################################################ Stage 2 #########################################
        # self.OClayer1_2 = nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        # self.OC1_bn_2 = nn.BatchNorm2d(256).requires_grad_(False)
        # self.OClayer2_2 = nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        # self.OC2_bn_2 = nn.BatchNorm2d(256).requires_grad_(False)
        # self.OClayer3_2 = nn.Conv2d(256,384,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        # self.OC3_bn_2 = nn.BatchNorm2d(384).requires_grad_(False)
        # self.OClayer4_2 = nn.Conv2d(384,512,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        # self.OC4_bn_2 = nn.BatchNorm2d(512).requires_grad_(False)

        # self.OCdeclayer1_2 = nn.Conv2d(512,384,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        # self.OC1_decbn_2 = nn.BatchNorm2d(384).requires_grad_(False)
        # self.OCdeclayer2_2 = nn.Conv2d(384,256,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        # self.OC2_decbn_2 = nn.BatchNorm2d(256).requires_grad_(False)
        # self.OCdeclayer3_2 = nn.Conv2d(256,256,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        # self.OC3_decbn_2 = nn.BatchNorm2d(256).requires_grad_(False)
        # self.OCdeclayer4_2 = nn.Conv2d(256,256,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        # self.OC4_decbn_2 = nn.BatchNorm2d(256).requires_grad_(False)

        # initialize_weights_kaimingnormal_forOC(self.OClayer1_2)
        # initialize_weights_kaimingnormal_forOC(self.OC1_bn_2)
        # initialize_weights_kaimingnormal_forOC(self.OClayer2_2)
        # initialize_weights_kaimingnormal_forOC(self.OC2_bn_2)
        # initialize_weights_kaimingnormal_forOC(self.OClayer3_2)
        # initialize_weights_kaimingnormal_forOC(self.OC3_bn_2)
        # initialize_weights_kaimingnormal_forOC(self.OClayer4_2)
        # initialize_weights_kaimingnormal_forOC(self.OC4_bn_2)
        # initialize_weights_kaimingnormal_forOC(self.OCdeclayer1_2)
        # initialize_weights_kaimingnormal_forOC(self.OC1_decbn_2)
        # initialize_weights_kaimingnormal_forOC(self.OCdeclayer2_2)
        # initialize_weights_kaimingnormal_forOC(self.OC2_decbn_2)
        # initialize_weights_kaimingnormal_forOC(self.OCdeclayer3_2)
        # initialize_weights_kaimingnormal_forOC(self.OC3_decbn_2)
        # initialize_weights_kaimingnormal_forOC(self.OCdeclayer4_2)
        # initialize_weights_kaimingnormal_forOC(self.OC4_decbn_2)
        # #########################################################################################################
    
        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        self.three_input_layer = False
        #self.IN = nn.InstanceNorm2d(3, affine=True)
        # self.xfm = DWTForward(J=6, mode='zero', wave='db3')
        # self.ixfm = DWTInverse(mode='zero', wave='db3')

    # def Normalization_Perturbation(self, feat):
    # # feat: input features of size (B, C, H, W)
    #     feat_mean = feat.mean((2, 3), keepdim=True) # size: B, C, 1, 1
    #     ones_mat = torch.ones_like(feat_mean)
    #     alpha = torch.normal(ones_mat, 0.1 * ones_mat) # size: B, C, 1, 1
    #     beta = torch.normal(ones_mat, 0.1 * ones_mat) # size: B, C, 1, 1
    #     output = alpha * feat - alpha * feat_mean + beta * feat_mean
    #     return output # size: B, C, H, W
    
    def Normalization_Perturbation_Plus(self, feat):
        feat_mean = feat.mean((2, 3), keepdim=True)
        ones_mat = torch.ones_like(feat_mean)
        zeros_mat = torch.zeros_like(feat_mean)
        mean_diff = torch.std(feat_mean, 0, keepdim=True)
        mean_scale = mean_diff / mean_diff.max() * 1.5
        alpha = torch.normal(ones_mat, 0.75 * ones_mat)
        beta = 1 + torch.normal(zeros_mat, 0.75 * ones_mat) * mean_scale
        output = alpha * feat - alpha * feat_mean + beta * feat_mean
        return output
    
    '''def Normalization_Perturbation_Plus_OC(self, feat):
        feat_mean = feat.mean((2, 3), keepdim=True)
        ones_mat = torch.ones_like(feat_mean)
        zeros_mat = torch.zeros_like(feat_mean)
        mean_diff = torch.std(feat_mean, 0, keepdim=True)
        mean_scale = mean_diff / mean_diff.max() * 1.5
        alpha = torch.normal(ones_mat, 1.0 * ones_mat)
        beta = 1 + torch.normal(zeros_mat, 1.0 * ones_mat) * mean_scale
        output = alpha * feat - alpha * feat_mean + beta * feat_mean
        return output'''
    

    def forward(self, x, gts=None, training=True):
        p = random.random()
        p2 = random.random()
        p3 = random.random()
        # p4 = random.random()
        w_arr = []
        x_hfi_bands = []
        x_size = x.size()  # 800
        h,w = x_size[2:]
        b,_,_,_ = x.shape

        if(training==True and p<0.5):
            initialize_weights_kaimingnormal_forOC(self.OClayer1)
            initialize_weights_kaimingnormal_forOC(self.OC1_bn)
            initialize_weights_kaimingnormal_forOC(self.OClayer2)
            initialize_weights_kaimingnormal_forOC(self.OC2_bn)
            initialize_weights_kaimingnormal_forOC(self.OClayer3)
            initialize_weights_kaimingnormal_forOC(self.OC3_bn)
            initialize_weights_kaimingnormal_forOC(self.OClayer4)
            initialize_weights_kaimingnormal_forOC(self.OC4_bn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
            initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
            initialize_weights_kaimingnormal_forOC(self.OC2_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer3)
            initialize_weights_kaimingnormal_forOC(self.OC3_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer4)
            initialize_weights_kaimingnormal_forOC(self.OC4_decbn)


            # initialize_weights_kaimingnormal_forOC(self.OClayer1_2)
            # initialize_weights_kaimingnormal_forOC(self.OC1_bn_2)
            # initialize_weights_kaimingnormal_forOC(self.OClayer2_2)
            # initialize_weights_kaimingnormal_forOC(self.OC2_bn_2)
            # initialize_weights_kaimingnormal_forOC(self.OClayer3_2)
            # initialize_weights_kaimingnormal_forOC(self.OC3_bn_2)
            # initialize_weights_kaimingnormal_forOC(self.OClayer4_2)
            # initialize_weights_kaimingnormal_forOC(self.OC4_bn_2)
            # initialize_weights_kaimingnormal_forOC(self.OCdeclayer1_2)
            # initialize_weights_kaimingnormal_forOC(self.OC1_decbn_2)
            # initialize_weights_kaimingnormal_forOC(self.OCdeclayer2_2)
            # initialize_weights_kaimingnormal_forOC(self.OC2_decbn_2)
            # initialize_weights_kaimingnormal_forOC(self.OCdeclayer3_2)
            # initialize_weights_kaimingnormal_forOC(self.OC3_decbn_2)
            # initialize_weights_kaimingnormal_forOC(self.OCdeclayer4_2)
            # initialize_weights_kaimingnormal_forOC(self.OC4_decbn_2)

           # ResNet
        x = self.layer0[0](x)
        #print(x.shape)
        x = self.layer0[1](x)
        #print(x.shape)
        x = self.layer0[2](x)
        #print(x.shape)
        x = self.layer0[3](x)
        xp = x
        if(training==True and p2<0.5):
            x = self.Normalization_Perturbation_Plus(xp)
            
        # OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp),scale_factor =(1.205,1.205))))
        # # if(training==True and p<0.5):
        # #     OCout = self.Normalization_Perturbation_Plus(OCout)
        # OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))
        # # if(training==True and p<0.5):
        # #     OCout = self.Normalization_Perturbation_Plus(OCout)
        # OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.2,1.2))))
        # # if(training==True and p<0.5):
        # #     OCout = self.Normalization_Perturbation_Plus(OCout)
        # OCout_dec = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), size =(int(h/2),int(w/2)))))
        # # if(training==True and p<0.5):
        # #     OCout = self.Normalization_Perturbation_Plus(OCout)

        OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp),scale_factor =(0.838,0.838))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(0.798,0.798))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(0.798,0.798))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout_dec = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), scale_factor =(0.869,0.869))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)

        # OCout = F.relu(self.OC1_decbn(F.interpolate(self.OCdeclayer1(OCout_dec), size =(int(h/2),int(w/2)))))
        OCout = F.relu(self.OC1_decbn(F.interpolate(self.OCdeclayer1(OCout_dec), scale_factor =(1.15,1.15))))
        OCout = F.relu(self.OC2_decbn(F.interpolate(self.OCdeclayer2(OCout), scale_factor =(1.2,1.2))))
        OCout = F.relu(self.OC3_decbn(F.interpolate(self.OCdeclayer3(OCout), scale_factor =(1.205,1.205))))
        OCout = F.relu(self.OC4_decbn(F.interpolate(self.OCdeclayer4(OCout), size =(math.ceil(h/4),math.ceil(w/4)))))
        ##GMU fusion##
        # hv = self.tanh(self.hv(x))
        # ht = self.tanh(self.ht(OCout))
        # z = self.sigmoid(self.z(torch.cat([x,OCout],dim=1)))
        # x = z*hv + (1-z)*ht
        if(training==True and p<0.5):
            x = torch.add(OCout, x)
        ##################
        x_tuple = self.layer1([x, w_arr])  # 400
        x_tuple_cp = x_tuple
        #f_map = x_tuple[0]
        if(training==True and p2<0.5):
            x_tuple[0] = self.Normalization_Perturbation_Plus(x_tuple[0])
        # hv = self.tanh(self.hv(x))
        # ht = self.tanh(self.ht(OCout))
        # z = self.sigmoid(self.z(torch.cat([x,OCout],dim=1)))
        # x = z*hv + (1-z)*ht

        low_level = x_tuple[0]
        #print(low_level.shape)
        x_tuple = self.layer2(x_tuple)  # 100
        #print(x_tuple[0].shape)
        x_tuple = self.layer3(x_tuple)  # 100
        ##print(x_tuple[0].shape)
        x_tuple = self.layer4(x_tuple)  # 100
        #print(x_tuple[0].shape)
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        
        x = x_tuple[0]
        w_arr = x_tuple[1]

        x = self.aspp(x)
        #print(x.shape)
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        dec0_up = self.bot_aspp(x)
        #print(dec0_up.shape)

        dec0_fine = self.bot_fine(low_level)
        #print(dec0_fine.shape)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        #print(dec0_up.shape)
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        #print(dec0.shape)
        dec1 = self.final1(dec0)
        #print(dec1.shape)
        # if(training==True and p3<0.5):
        #     dec1 = Upsample(dec1, (int(h/2),int(w/2)))
        #     dec1 = torch.add(OCout_dec, dec1)
        
        #dec2 = torch.add(OCout, dec2)
        dec2 = self.final2(dec1)
        #print(dec2.shape)
        main_out = Upsample(dec2, x_size[2:])
        #print(main_out.shape)
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        #print(hey)
        if training:
            loss1 = self.criterion(main_out, gts)
            return_loss = loss1
            return return_loss
        else:
            return main_out
        

class LabMAO_MAOEncPerturb_noskipconn_randomOC_godmodel_INAffineTrue_perturbindec(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,4,4,4,0,0], use_wtloss=False):
        super(LabMAO_MAOEncPerturb_noskipconn_randomOC_godmodel_INAffineTrue_perturbindec, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.variant = variant
        self.wt_layer = wt_layer
        self.use_wtloss = use_wtloss
        self.trunk = trunk
        
        channel_3rd = 256
        prev_final_channel = 1024
        final_channel = 2048
        
        if trunk == 'resnet-50':
            resnet = Resnet.resnet50(wt_layer=self.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            # raise 'unknown deepv3 variant: {}'.format(self.variant)
            print("Not using Dilation ")

        os=16  ######### D16 ###############

        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                    output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        self.OClayer1 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC1_bn = nn.BatchNorm2d(64).requires_grad_(False)
        #self.OC1_IN = nn.InstanceNorm2d(64,affine=False).requires_grad_(False)
        self.OClayer2 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC2_bn = nn.BatchNorm2d(64).requires_grad_(False)
        #self.OC2_IN = nn.InstanceNorm2d(64,affine=False).requires_grad_(False)
        self.OClayer3 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC3_bn = nn.BatchNorm2d(128).requires_grad_(False)
        #self.OC3_IN = nn.InstanceNorm2d(128,affine=False).requires_grad_(False)
        self.OClayer4 = nn.Conv2d(128,256,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC4_bn = nn.BatchNorm2d(256).requires_grad_(False)

        self.OCdeclayer1 = nn.Conv2d(256,128,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC1_decbn = nn.BatchNorm2d(128).requires_grad_(False)
        self.OCdeclayer2 = nn.Conv2d(128,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC2_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC3_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer4 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC4_decbn = nn.BatchNorm2d(64).requires_grad_(False)

        initialize_weights_kaimingnormal_forOC(self.OClayer1)
        initialize_weights_kaimingnormal_forOC(self.OC1_bn)
        #initialize_weights_kaimingnormal_forOC(self.OC1_IN)
        initialize_weights_kaimingnormal_forOC(self.OClayer2)
        initialize_weights_kaimingnormal_forOC(self.OC2_bn)
        #initialize_weights_kaimingnormal_forOC(self.OC2_IN)
        initialize_weights_kaimingnormal_forOC(self.OClayer3)
        initialize_weights_kaimingnormal_forOC(self.OC3_bn)
        #initialize_weights_kaimingnormal_forOC(self.OC3_IN)
        initialize_weights_kaimingnormal_forOC(self.OClayer4)
        initialize_weights_kaimingnormal_forOC(self.OC4_bn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
        initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
        initialize_weights_kaimingnormal_forOC(self.OC2_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer3)
        initialize_weights_kaimingnormal_forOC(self.OC3_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer4)
        initialize_weights_kaimingnormal_forOC(self.OC4_decbn)
    
        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        self.three_input_layer = False
        # self.xfm = DWTForward(J=6, mode='zero', wave='db3')
        # self.ixfm = DWTInverse(mode='zero', wave='db3')

    # def Normalization_Perturbation(self, feat):
    # # feat: input features of size (B, C, H, W)
    #     feat_mean = feat.mean((2, 3), keepdim=True) # size: B, C, 1, 1
    #     ones_mat = torch.ones_like(feat_mean)
    #     alpha = torch.normal(ones_mat, 0.1 * ones_mat) # size: B, C, 1, 1
    #     beta = torch.normal(ones_mat, 0.1 * ones_mat) # size: B, C, 1, 1
    #     output = alpha * feat - alpha * feat_mean + beta * feat_mean
    #     return output # size: B, C, H, W
    
    def Normalization_Perturbation_Plus(self, feat):
        feat_mean = feat.mean((2, 3), keepdim=True)
        ones_mat = torch.ones_like(feat_mean)
        zeros_mat = torch.zeros_like(feat_mean)
        mean_diff = torch.std(feat_mean, 0, keepdim=True)
        mean_scale = mean_diff / mean_diff.max() * 1.5
        alpha = torch.normal(ones_mat, 0.75 * ones_mat)
        beta = 1 + torch.normal(zeros_mat, 0.75 * ones_mat) * mean_scale
        output = alpha * feat - alpha * feat_mean + beta * feat_mean
        return output
    
    '''def Normalization_Perturbation_Plus_OC(self, feat):
        feat_mean = feat.mean((2, 3), keepdim=True)
        ones_mat = torch.ones_like(feat_mean)
        zeros_mat = torch.zeros_like(feat_mean)
        mean_diff = torch.std(feat_mean, 0, keepdim=True)
        mean_scale = mean_diff / mean_diff.max() * 1.5
        alpha = torch.normal(ones_mat, 1.0 * ones_mat)
        beta = 1 + torch.normal(zeros_mat, 1.0 * ones_mat) * mean_scale
        output = alpha * feat - alpha * feat_mean + beta * feat_mean
        return output'''
    

    def forward(self, x, gts=None, training=True):
        # print(x.shape)
        p = random.random()
        p2 = random.random()
        p3 = random.random()
        w_arr = []
        #x_hfi_bands = []
        x_size = x.size()  # 800
        h,w = x_size[2:]
        b,_,_,_ = x.shape


        if(training==True):
            initialize_weights_kaimingnormal_forOC(self.OClayer1)
            initialize_weights_kaimingnormal_forOC(self.OC1_bn)
            #initialize_weights_kaimingnormal_forOC(self.OC1_IN)
            initialize_weights_kaimingnormal_forOC(self.OClayer2)
            initialize_weights_kaimingnormal_forOC(self.OC2_bn)
            #initialize_weights_kaimingnormal_forOC(self.OC2_IN)
            initialize_weights_kaimingnormal_forOC(self.OClayer3)
            initialize_weights_kaimingnormal_forOC(self.OC3_bn)
            #initialize_weights_kaimingnormal_forOC(self.OC3_IN)
            initialize_weights_kaimingnormal_forOC(self.OClayer4)
            initialize_weights_kaimingnormal_forOC(self.OC4_bn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
            initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
            initialize_weights_kaimingnormal_forOC(self.OC2_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer3)
            initialize_weights_kaimingnormal_forOC(self.OC3_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer4)
            initialize_weights_kaimingnormal_forOC(self.OC4_decbn)

           # ResNet
        x = self.layer0[0](x)
        #print(x.shape)
        x = self.layer0[1](x)
        #print(x.shape)
        x = self.layer0[2](x)
        #print(x.shape)
        x = self.layer0[3](x)
        xp = x
        # print(xp.shape)
        if(training==True and p2<0.5):
            x = self.Normalization_Perturbation_Plus(xp)
        # xNP = x
            # print(x)
            # print(hey)
            
        OCout1 = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp),scale_factor =(1.205,1.205))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        # OCout2 = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout1), scale_factor =(1.2,1.2))))
        OCout2 = F.relu(F.interpolate(self.OClayer2(OCout1), scale_factor =(1.2,1.2)))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout3 = F.relu(F.interpolate(self.OClayer3(OCout2), scale_factor =(1.2,1.2)))
        # OCout3 = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout2), scale_factor =(1.2,1.2))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout_dec = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout3), size =(int(h/2),int(w/2)))))
        # print(OCout_dec.shape)
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)

        OCoutdeclayer1 = F.relu(self.OC1_decbn(F.interpolate(self.OCdeclayer1(OCout_dec), size =(int(h/2),int(w/2)))))
        OCoutdeclayer2 = F.relu(self.OC2_decbn(F.interpolate(self.OCdeclayer2(OCoutdeclayer1), scale_factor =(0.838,0.838))))
        OCoutdeclayer3 = F.relu(self.OC3_decbn(F.interpolate(self.OCdeclayer3(OCoutdeclayer2), scale_factor =(0.798,0.798))))
        OCout = F.relu(self.OC4_decbn(F.interpolate(self.OCdeclayer4(OCoutdeclayer3), size =(math.ceil(h/4),math.ceil(w/4)))))

        if(training==True and p<0.5):
            x = torch.add(OCout, x)
        ##################
        x_tuple = self.layer1([x, w_arr])  # 400
        #f_map = x_tuple[0]
        if(training==True and p2<0.5):
            x_tuple[0] = self.Normalization_Perturbation_Plus(x_tuple[0])
        # xNP = x_tuple[0]
        low_level = x_tuple[0]
        #print(low_level.shape)
        x_tuple = self.layer2(x_tuple)  # 100
        #print(x_tuple[0].shape)
        x_tuple = self.layer3(x_tuple)  # 100
        ##print(x_tuple[0].shape)
        x_tuple = self.layer4(x_tuple)  # 100
        #print(x_tuple[0].shape)
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        # x_tsne = x_tuple[0]
        x = x_tuple[0]
        w_arr = x_tuple[1]

        x = self.aspp(x)
        #print(x.shape)
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        dec0_up = self.bot_aspp(x)
        #print(dec0_up.shape)

        dec0_fine = self.bot_fine(low_level)
        #print(dec0_fine.shape)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        #print(dec0_up.shape)
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        #print(dec0.shape)
        dec1 = self.final1(dec0)
        #print(dec1.shape)
        #dec1 = Upsample(dec1, (int(h/2),int(w/2)))
        if(training==True and p3<0.5):
            dec1 = Upsample(dec1, (int(h/2),int(w/2)))
            dec1 = torch.add(OCout_dec, dec1)

        
        #dec2 = torch.add(OCout, dec2)
        dec2 = self.final2(dec1)
        #print(dec2.shape)
        main_out = Upsample(dec2, x_size[2:])
        #print(main_out.shape)
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        #print(hey)
        if training:
            loss1 = self.criterion(main_out, gts)
            return_loss = loss1
            return return_loss
        else:
            return main_out


class MRFPPlus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,4,4,4,0,0], use_wtloss=False):
        super(MRFPPlus, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.variant = variant
        self.wt_layer = wt_layer
        self.use_wtloss = use_wtloss
        self.trunk = trunk
        
        channel_3rd = 256
        prev_final_channel = 1024
        final_channel = 2048
        
        if trunk == 'resnet-50':
            resnet = Resnet.resnet50(wt_layer=self.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            # raise 'unknown deepv3 variant: {}'.format(self.variant)
            print("Not using Dilation ")

        os=16  ######### D16 ###############

        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                    output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        self.OClayer1 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC1_bn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OClayer2 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC2_bn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OClayer3 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC3_bn = nn.BatchNorm2d(128).requires_grad_(False)
        self.OClayer4 = nn.Conv2d(128,256,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC4_bn = nn.BatchNorm2d(256).requires_grad_(False)

        self.OCdeclayer1 = nn.Conv2d(256,128,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC1_decbn = nn.BatchNorm2d(128).requires_grad_(False)
        self.OCdeclayer2 = nn.Conv2d(128,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC2_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC3_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer4 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC4_decbn = nn.BatchNorm2d(64).requires_grad_(False)

        initialize_weights_kaimingnormal_forOC(self.OClayer1)
        initialize_weights_kaimingnormal_forOC(self.OC1_bn)
        initialize_weights_kaimingnormal_forOC(self.OClayer2)
        initialize_weights_kaimingnormal_forOC(self.OC2_bn)
        initialize_weights_kaimingnormal_forOC(self.OClayer3)
        initialize_weights_kaimingnormal_forOC(self.OC3_bn)
        initialize_weights_kaimingnormal_forOC(self.OClayer4)
        initialize_weights_kaimingnormal_forOC(self.OC4_bn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
        initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
        initialize_weights_kaimingnormal_forOC(self.OC2_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer3)
        initialize_weights_kaimingnormal_forOC(self.OC3_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer4)
        initialize_weights_kaimingnormal_forOC(self.OC4_decbn)
    
        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        self.three_input_layer = False
    
    def Normalization_Perturbation_Plus(self, feat):
        feat_mean = feat.mean((2, 3), keepdim=True)
        ones_mat = torch.ones_like(feat_mean)
        zeros_mat = torch.zeros_like(feat_mean)
        mean_diff = torch.std(feat_mean, 0, keepdim=True)
        mean_scale = mean_diff / mean_diff.max() * 1.5
        alpha = torch.normal(ones_mat, 0.75 * ones_mat)
        beta = 1 + torch.normal(zeros_mat, 0.75 * ones_mat) * mean_scale
        output = alpha * feat - alpha * feat_mean + beta * feat_mean
        return output
    

    def forward(self, x, gts=None, training=True):
        p = random.random()
        p2 = random.random()
        p3 = random.random()
        w_arr = []
        x_size = x.size()  # 800
        h,w = x_size[2:]
        b,_,_,_ = x.shape


        if(training==True and p<0.5):
            initialize_weights_kaimingnormal_forOC(self.OClayer1)
            initialize_weights_kaimingnormal_forOC(self.OC1_bn)
            initialize_weights_kaimingnormal_forOC(self.OClayer2)
            initialize_weights_kaimingnormal_forOC(self.OC2_bn)
            initialize_weights_kaimingnormal_forOC(self.OClayer3)
            initialize_weights_kaimingnormal_forOC(self.OC3_bn)
            initialize_weights_kaimingnormal_forOC(self.OClayer4)
            initialize_weights_kaimingnormal_forOC(self.OC4_bn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
            initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
            initialize_weights_kaimingnormal_forOC(self.OC2_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer3)
            initialize_weights_kaimingnormal_forOC(self.OC3_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer4)
            initialize_weights_kaimingnormal_forOC(self.OC4_decbn)

           # ResNet
        x = self.layer0[0](x)

        x = self.layer0[1](x)

        x = self.layer0[2](x)

        x = self.layer0[3](x)
        xp = x
        if(training==True and p2<0.5):
            x = self.Normalization_Perturbation_Plus(xp)
            
        OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp),scale_factor =(1.205,1.205))))
        OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))
        OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.2,1.2))))
        OCout_dec = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), size =(int(h/2),int(w/2)))))
        OCout = F.relu(self.OC1_decbn(F.interpolate(self.OCdeclayer1(OCout_dec), size =(int(h/2),int(w/2)))))
        OCout = F.relu(self.OC2_decbn(F.interpolate(self.OCdeclayer2(OCout), scale_factor =(0.838,0.838))))
        OCout = F.relu(self.OC3_decbn(F.interpolate(self.OCdeclayer3(OCout), scale_factor =(0.798,0.798))))
        OCout = F.relu(self.OC4_decbn(F.interpolate(self.OCdeclayer4(OCout), size =(math.ceil(h/4),math.ceil(w/4)))))

        if(training==True and p<0.5):
            x = torch.add(OCout, x)

        x_tuple = self.layer1([x, w_arr])  # 400

        if(training==True and p2<0.5):
            x_tuple[0] = self.Normalization_Perturbation_Plus(x_tuple[0])
        low_level = x_tuple[0]

        x_tuple = self.layer2(x_tuple)  # 100

        x_tuple = self.layer3(x_tuple)  # 100

        x_tuple = self.layer4(x_tuple)  # 100
        
        x = x_tuple[0]
        w_arr = x_tuple[1]

        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)

        dec0_fine = self.bot_fine(low_level)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final1(dec0)
        if(training==True and p3<0.5):
            dec1 = Upsample(dec1, (int(h/2),int(w/2)))
            dec1 = torch.add(OCout_dec, dec1)

        
        dec2 = self.final2(dec1)
        main_out = Upsample(dec2, x_size[2:])
        if training:
            loss1 = self.criterion(main_out, gts)
            return_loss = loss1
            return return_loss
        else:
            return main_out


class simpleDeepV3Plus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,0,0,0,0,0], use_wtloss=False):
        super(simpleDeepV3Plus, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.variant = variant
        self.wt_layer = wt_layer
        self.use_wtloss = use_wtloss
        self.trunk = trunk
        
        channel_3rd = 256
        prev_final_channel = 1024
        final_channel = 2048
        
        if trunk == 'resnet-50':
            resnet = Resnet.resnet50(wt_layer=self.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            # raise 'unknown deepv3 variant: {}'.format(self.variant)
            print("Not using Dilation ")

        os=16  ######### D16 ###############

        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                    output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        self.three_input_layer = False

    def forward(self, x, gts=None,training=False):
        w_arr = []
        x_size = x.size()  # 800

        # ResNet
        x = self.layer0[0](x)
        x = self.layer0[1](x)
        x = self.layer0[2](x)
        x = self.layer0[3](x)


        x_tuple = self.layer1([x, w_arr])  # 400


        low_level = x_tuple[0]
        x_tuple = self.layer2(x_tuple)  # 100
        x_tuple = self.layer3(x_tuple)  # 100
        x_tuple = self.layer4(x_tuple)  # 100
        
        x = x_tuple[0]
        w_arr = x_tuple[1]

        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)

        dec0_fine = self.bot_fine(low_level)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final1(dec0)
        dec2 = self.final2(dec1)

        main_out = Upsample(dec2, x_size[2:])

        if training:
            loss1 = self.criterion(main_out, gts)
            return_loss = loss1
            return return_loss
        else:
            return main_out