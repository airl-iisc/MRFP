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

class LabMAO_wirin(nn.Module):
    """
    includes NP
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,0,0,0,0,0], use_wtloss=False):
        super(LabMAO_wirin, self).__init__()
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


        self.OClayer1 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)
        self.OC1_bn = nn.BatchNorm2d(64)
        self.OClayer2 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)
        self.OC2_bn = nn.BatchNorm2d(64)
        self.OClayer3 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC3_bn = nn.BatchNorm2d(128)
        self.OClayer4 = nn.Conv2d(128,256,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC4_bn = nn.BatchNorm2d(256)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hv = nn.Conv2d(256,256,kernel_size=1,stride=1,padding=1)
        self.ht = nn.Conv2d(256,256,kernel_size=1,stride=1,padding=1)
        self.z = nn.Conv2d(512,256,kernel_size=1,stride=1,padding=1)


        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        self.three_input_layer = False

        self.xfm = DWTForward(J=3, mode='zero', wave='db3')
        self.ixfm = DWTInverse(mode='zero', wave='db3')

    def Normalization_Perturbation(self, feat):
    # feat: input features of size (B, C, H, W)
        feat_mean = feat.mean((2, 3), keepdim=True) # size: B, C, 1, 1
        ones_mat = torch.ones_like(feat_mean)
        alpha = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
        beta = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
        output = alpha * feat - alpha * feat_mean + beta * feat_mean
        return output # size: B, C, H, W

    def forward(self, x, gts=None, training=True):
        p = random.random()
        w_arr = []
        x_hfl = []
        x_size = x.size()  # 800
        h,w = x_size[2:]
        #print(x_size)
        #print(x.shape)
           # ResNet
        x = self.layer0[0](x)
        #print(x.shape)
        x = self.layer0[1](x)
        #print(x.shape)
        x = self.layer0[2](x)
        #print(x.shape)
        x = self.layer0[3](x)
        xp = x
        ########################### DWT ##################################
        '''
        x_lf,x_hf = self.xfm(x)
        for i in range(len(x_hf)):
            x_hfl.append(torch.zeros(x_hf[i].size()).cuda())
        #x_hfl.append(torch.zeros(x_hf[0].size()).cuda())
        #x_hfl.append(x_hf[1])
        #x_hfl.append(x_hf[2])
        #x_lfl = torch.zeros(x_lf.size()).cuda()
        #x_hf = self.ixfm((x_lfl,x_hfl))
        x_lf = self.ixfm((x_lf,x_hfl))
        #x_mid = self.ixfm((x_lfl,x_hfl))
        '''
        ##################################################################
        if(training==True and p<0.5):
            #print('hello - --------train')
            x = self.Normalization_Perturbation(xp)     ##Is this input correct???
        OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp),scale_factor =(1.205,1.205)))) #layersize256 #output320   ##x_lf = 
        if(training==True and p<0.5):
            OCout = self.Normalization_Perturbation(OCout)
        OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))#layersize320 #output400
        if(training==True and p<0.5):
            OCout = self.Normalization_Perturbation(OCout)
        OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.2,1.2))))#layersize400 output500
        #print(OCout.shape)
        OCout = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), size =(int(h/2),int(w/2)))))#layersize500 output625

        x_tuple = self.layer1([x, w_arr])  # 400
        if(training==True and p<0.5):
            x_tuple[0] = self.Normalization_Perturbation(x_tuple[0])
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
        dec1 = Upsample(dec1, (int(h/2),int(w/2)))
        #print('dec1----',dec1.shape)
        #print('OCout----',OCout.shape)
        ##Tune the inputs to GMU block.
        #dec1 = dec1 + OCout ##updated part.
        #OCout = dec1 + OCout
        dec1 = dec1 + OCout  ##dec1 is updated
        #print('U.C stays as it is, o.c is updated')

        hv = self.tanh(self.hv(dec1))
        ht = self.tanh(self.ht(OCout))
        z = self.sigmoid(self.z(torch.cat([dec1,OCout],dim=1)))
        dec1 = z*hv + (1-z)*ht
        
        #dec2 = torch.add(OCout, dec2)
        dec2 = self.final2(dec1)
        
        #print(dec2.shape)
        main_out = Upsample(dec2, x_size[2:])
        
        #print(main_out.shape)
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        #print(hey)
        #
        if training:
            
            loss1 = self.criterion(main_out, gts)
            return_loss = loss1
            return return_loss
        else:
            #print('main_out',main_out)
            return main_out
        
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

class LabMAO(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,0,0,0,0,0], use_wtloss=False):
        super(LabMAO, self).__init__()
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

        self.OClayer1 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)
        self.OC1_bn = nn.BatchNorm2d(64)
        self.OClayer2 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)
        self.OC2_bn = nn.BatchNorm2d(64)
        self.OClayer3 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC3_bn = nn.BatchNorm2d(128)
        self.OClayer4 = nn.Conv2d(128,256,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC4_bn = nn.BatchNorm2d(256)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hv = nn.Conv2d(256,256,kernel_size=1,stride=1,padding=1)
        self.ht = nn.Conv2d(256,256,kernel_size=1,stride=1,padding=1)
        self.z = nn.Conv2d(512,256,kernel_size=1,stride=1,padding=1)

        #self.GRL = GradientReversal_Layer(alpha=1)

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        self.three_input_layer = False

        self.xfm = DWTForward(J=6, mode='zero', wave='db6')
        self.ixfm = DWTInverse(mode='zero', wave='db6')

    # def Normalization_Perturbation(self, feat):
    # # feat: input features of size (B, C, H, W)
    #     feat_mean = feat.mean((2, 3), keepdim=True) # size: B, C, 1, 1
    #     ones_mat = torch.ones_like(feat_mean)
    #     alpha = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
    #     beta = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
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
    
    def Normalization_Perturbation_Plus_OC(self, feat):
        feat_mean = feat.mean((2, 3), keepdim=True)
        ones_mat = torch.ones_like(feat_mean)
        zeros_mat = torch.zeros_like(feat_mean)
        mean_diff = torch.std(feat_mean, 0, keepdim=True)
        mean_scale = mean_diff / mean_diff.max() * 1.5
        alpha = torch.normal(ones_mat, 1.0 * ones_mat)
        beta = 1 + torch.normal(zeros_mat, 1.0 * ones_mat) * mean_scale
        output = alpha * feat - alpha * feat_mean + beta * feat_mean
        return output

    def forward(self, x, gts=None, training=True):
        p = random.random()
        w_arr = []
        x_hfl = []
        x_midf = []
        x_size = x.size()  # 800
        h,w = x_size[2:]
        #print(x_size)
        #print(x.shape)
        # ########################### DWT ##################################
        # x_lf,x_hf = self.xfm(x)
        # #for i in range(len(x_hf)):
        # #   x_hfl.append(torch.zeros(x_hf[i].size()).cuda())
        # x_midf.append(torch.zeros(x_hf[0].size()).cuda())
        # #x_hfl.append(x_hf[0])
        # x_midf.append(x_hf[1])
        # x_midf.append(x_hf[2])
        # x_midf.append(x_hf[3])
        # x_midf.append(x_hf[4])
        # x_midf.append(x_hf[5])

        # x_hfl.append(torch.zeros(x_hf[0].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[1].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[2].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[3].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[4].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[5].size()).cuda())

        # x_lfl = torch.zeros(x_lf.size()).cuda()
        # x_mid = self.ixfm((x_lfl,x_midf))
        # #x_hf = self.ixfm((x_lfl,x_hf))
        # #x_lf = self.ixfm((x_lf,x_hfl))
        # #x_mid = self.ixfm((x_lfl,x_hfl))
        # x_lfi = self.ixfm((x_lf,x_hfl))
        # x_lfi = F.interpolate(x_lfi, size=(192,192))
        # ##################################################################
           # ResNet
        x = self.layer0[0](x)
        #print(x.shape)
        x = self.layer0[1](x)
        #print(x.shape)
        x = self.layer0[2](x)
        #print(x.shape)
        x = self.layer0[3](x)
        #xp = x

        ########################### DWT ##################################
        x_lf,x_hf = self.xfm(x)
        #for i in range(len(x_hf)):
        #   x_hfl.append(torch.zeros(x_hf[i].size()).cuda())
        #x_midf.append(torch.zeros(x_hf[0].size()).cuda())
        #x_hfl.append(x_hf[0])
        x_midf.append(x_hf[0])
        x_midf.append(x_hf[1])
        x_midf.append(x_hf[2])
        x_midf.append(x_hf[3])
        x_midf.append(x_hf[4])
        x_midf.append(x_hf[5])

        x_hfl.append(torch.zeros(x_hf[0].size()).cuda())
        x_hfl.append(torch.zeros(x_hf[1].size()).cuda())
        x_hfl.append(torch.zeros(x_hf[2].size()).cuda())
        x_hfl.append(torch.zeros(x_hf[3].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[4].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[5].size()).cuda())
        x_hfl.append(x_hf[4])
        x_hfl.append(x_hf[5])

        x_lfl = torch.zeros(x_lf.size()).cuda()
        x_UC = self.ixfm((x_lfl,x_midf))
        #x_hf = self.ixfm((x_lfl,x_hf))
        x_OC = self.ixfm((x_lf,x_hfl))
        #x_mid = self.ixfm((x_lfl,x_hfl))
        #x_hlf = self.ixfm((x_lfl,x_hfl))
        #x_hlf = F.interpolate(x_hlf, size=(192,192))

        ##################################################################

        #x = x_midlf
        if(training==True and p<0.5):
            x = self.Normalization_Perturbation_Plus(x_UC)
        OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(x_OC),scale_factor =(1.205,1.205)))) #layersize256 #output320
        if(training==True and p<0.5):
            OCout = self.Normalization_Perturbation_Plus_OC(OCout)
        OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))#layersize320 #output400
        if(training==True and p<0.5):
            OCout = self.Normalization_Perturbation_Plus_OC(OCout)
        OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.2,1.2))))#layersize400 output500
        #print(OCout.shape)
        OCout = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), size =(int(h/2),int(w/2)))))#layersize500 output625

        x_tuple = self.layer1([x, w_arr])  # 400
        #f_map = x_tuple[0]
        if(training==True and p<0.5):
            x_tuple[0] = self.Normalization_Perturbation_Plus(x_tuple[0])
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
        dec1 = Upsample(dec1, (int(h/2),int(w/2)))

        hv = self.tanh(self.hv(dec1))
        ht = self.tanh(self.ht(OCout))
        z = self.sigmoid(self.z(torch.cat([dec1,OCout],dim=1)))
        dec1 = z*hv + (1-z)*ht
        
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

class LabMAO_MAOEncPerturb(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,0,0,0,0,0], use_wtloss=False):
        super(LabMAO_MAOEncPerturb, self).__init__()
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

        self.OClayer1 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)
        self.OC1_bn = nn.BatchNorm2d(64)
        self.OClayer2 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)
        self.OC2_bn = nn.BatchNorm2d(64)
        self.OClayer3 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC3_bn = nn.BatchNorm2d(128)
        self.OClayer4 = nn.Conv2d(128,256,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC4_bn = nn.BatchNorm2d(256)

        self.OCdeclayer1 = nn.Conv2d(256,128,kernel_size=3, stride=1, padding=1)
        self.OC1_decbn = nn.BatchNorm2d(128)
        self.OCdeclayer2 = nn.Conv2d(128,64,kernel_size=3, stride=1, padding=1)
        self.OC2_decbn = nn.BatchNorm2d(64)
        self.OCdeclayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC3_decbn = nn.BatchNorm2d(64)
        self.OCdeclayer4 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC4_decbn = nn.BatchNorm2d(64)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hv = nn.Conv2d(64,64,kernel_size=1,stride=1,padding=1)
        self.ht = nn.Conv2d(64,64,kernel_size=1,stride=1,padding=1)
        self.z = nn.Conv2d(128,64,kernel_size=1,stride=1,padding=1)

        #self.GRL = GradientReversal_Layer(alpha=1)

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        self.three_input_layer = False

        #self.xfm = DWTForward(J=3, mode='zero', wave='db3')
        #self.ixfm = DWTInverse(mode='zero', wave='db3')

    # def Normalization_Perturbation(self, feat):
    # # feat: input features of size (B, C, H, W)
    #     feat_mean = feat.mean((2, 3), keepdim=True) # size: B, C, 1, 1
    #     ones_mat = torch.ones_like(feat_mean)
    #     alpha = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
    #     beta = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
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
    
    def Normalization_Perturbation_Plus_OC(self, feat):
        feat_mean = feat.mean((2, 3), keepdim=True)
        ones_mat = torch.ones_like(feat_mean)
        zeros_mat = torch.zeros_like(feat_mean)
        mean_diff = torch.std(feat_mean, 0, keepdim=True)
        mean_scale = mean_diff / mean_diff.max() * 1.5
        alpha = torch.normal(ones_mat, 1.0 * ones_mat)
        beta = 1 + torch.normal(zeros_mat, 1.0 * ones_mat) * mean_scale
        output = alpha * feat - alpha * feat_mean + beta * feat_mean
        return output

    def forward(self, x, gts=None, training=True):
        p = random.random()
        w_arr = []
        x_hfl = []
        x_midf = []
        x_size = x.size()  # 800
        h,w = x_size[2:]
        #print(x_size)
        #print(x.shape)
        # ########################### DWT ##################################
        # x_lf,x_hf = self.xfm(x)
        # #for i in range(len(x_hf)):
        # #   x_hfl.append(torch.zeros(x_hf[i].size()).cuda())
        # x_midf.append(torch.zeros(x_hf[0].size()).cuda())
        # #x_hfl.append(x_hf[0])
        # x_midf.append(x_hf[1])
        # x_midf.append(x_hf[2])
        # x_midf.append(x_hf[3])
        # x_midf.append(x_hf[4])
        # x_midf.append(x_hf[5])

        # x_hfl.append(torch.zeros(x_hf[0].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[1].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[2].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[3].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[4].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[5].size()).cuda())

        # x_lfl = torch.zeros(x_lf.size()).cuda()
        # x_mid = self.ixfm((x_lfl,x_midf))
        # #x_hf = self.ixfm((x_lfl,x_hf))
        # #x_lf = self.ixfm((x_lf,x_hfl))
        # #x_mid = self.ixfm((x_lfl,x_hfl))
        # x_lfi = self.ixfm((x_lf,x_hfl))
        # x_lfi = F.interpolate(x_lfi, size=(192,192))
        # ##################################################################
           # ResNet
        x = self.layer0[0](x)
        #print(x.shape)
        x = self.layer0[1](x)
        #print(x.shape)
        x = self.layer0[2](x)
        #print(x.shape)
        x = self.layer0[3](x)
        xp = x

        # ########################### DWT ##################################
        # x_lf,x_hf = self.xfm(x)
        # #for i in range(len(x_hf)):
        # #   x_hfl.append(torch.zeros(x_hf[i].size()).cuda())
        # x_midf.append(torch.zeros(x_hf[0].size()).cuda())
        # #x_hfl.append(x_hf[0])
        # x_midf.append(x_hf[1])
        # x_midf.append(x_hf[2])
        # # x_midf.append(x_hf[3])
        # # x_midf.append(x_hf[4])
        # # x_midf.append(x_hf[5])

        # x_hfl.append(torch.zeros(x_hf[0].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[1].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[2].size()).cuda())
        # # x_hfl.append(torch.zeros(x_hf[3].size()).cuda())
        # # x_hfl.append(torch.zeros(x_hf[4].size()).cuda())
        # # x_hfl.append(torch.zeros(x_hf[5].size()).cuda())

        # #x_lfl = torch.zeros(x_lf.size()).cuda()
        # x_midlf = self.ixfm((x_lf,x_midf))
        # #x_hf = self.ixfm((x_lfl,x_hf))
        # x_lfi = self.ixfm((x_lf,x_hfl))
        # #x_mid = self.ixfm((x_lfl,x_hfl))
        # #x_hlf = self.ixfm((x_lfl,x_hfl))
        # #x_hlf = F.interpolate(x_hlf, size=(192,192))
        # ##################################################################

        #x = x_midlf
        if(training==True and p<0.5):
            x = self.Normalization_Perturbation_Plus(xp)
        OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp),scale_factor =(1.205,1.205))))
        if(training==True and p<0.5):
            OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))
        if(training==True and p<0.5):
            OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.2,1.2))))
        OCout = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), size =(int(h/2),int(w/2)))))

        OCout = F.relu(self.OC1_decbn(F.interpolate(self.OCdeclayer1(OCout), size =(int(h/2),int(w/2)))))
        OCout = F.relu(self.OC2_decbn(F.interpolate(self.OCdeclayer2(OCout), scale_factor =(0.838,0.838))))
        OCout = F.relu(self.OC3_decbn(F.interpolate(self.OCdeclayer3(OCout), scale_factor =(0.798,0.798))))
        OCout = F.relu(self.OC4_decbn(F.interpolate(self.OCdeclayer4(OCout), size =(int(h/4),int(w/4)))))

        ##GMU fusion##
        hv = self.tanh(self.hv(x))
        ht = self.tanh(self.ht(OCout))
        z = self.sigmoid(self.z(torch.cat([x,OCout],dim=1)))
        x = z*hv + (1-z)*ht
        ##################
        x_tuple = self.layer1([x, w_arr])  # 400
        #f_map = x_tuple[0]
        if(training==True and p<0.5):
            x_tuple[0] = self.Normalization_Perturbation_Plus(x_tuple[0])
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
        #dec1 = Upsample(dec1, (int(h/2),int(w/2)))


        
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

class Lab2UC(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,0,0,0,0,0], use_wtloss=False):
        super(Lab2UC, self).__init__()
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
        
        # self.dsn = nn.Sequential(
        #     nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
        #     Norm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(0.1),
        #     nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # )

        self.OClayer1 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)
        self.OC1_bn = nn.BatchNorm2d(64)
        self.OClayer2 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)
        self.OC2_bn = nn.BatchNorm2d(64)
        self.OClayer3 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC3_bn = nn.BatchNorm2d(128)
        self.OClayer4 = nn.Conv2d(128,256,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC4_bn = nn.BatchNorm2d(256)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hv = nn.Conv2d(256,256,kernel_size=1,stride=1,padding=1)

        self.ht = nn.Conv2d(256,256,kernel_size=1,stride=1,padding=1)

        self.z = nn.Conv2d(512,256,kernel_size=1,stride=1,padding=1)

        #initialize_weights(self.dsn)
        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        self.three_input_layer = False

        self.xfm = DWTForward(J=6, mode='zero', wave='db3')
        self.ixfm = DWTInverse(mode='zero', wave='db3')

    def Normalization_Perturbation(self, feat):
    # feat: input features of size (B, C, H, W)
        feat_mean = feat.mean((2, 3), keepdim=True) # size: B, C, 1, 1
        ones_mat = torch.ones_like(feat_mean)
        alpha = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
        beta = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
        output = alpha * feat - alpha * feat_mean + beta * feat_mean
        return output # size: B, C, H, W
    
    def Normalization_Perturbation_Plus_mid(self, feat):
        feat_mean = feat.mean((2, 3), keepdim=True)
        ones_mat = torch.ones_like(feat_mean)
        zeros_mat = torch.zeros_like(feat_mean)
        mean_diff = torch.std(feat_mean, 0, keepdim=True)
        mean_scale = mean_diff / mean_diff.max() * 1.5
        alpha = torch.normal(ones_mat, 0.5 * ones_mat)
        beta = 1 + torch.normal(zeros_mat, 0.5 * ones_mat) * mean_scale
        output = alpha * feat - alpha * feat_mean + beta * feat_mean
        return output
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
        w_arr = []
        x_midf = []
        x_hfl = []
        x_size = x.size()  # 800
        h,w = x_size[2:]
        #print(x_size)
        #print(x.shape)
        ########################### DWT ##################################
        x_lf,x_hf = self.xfm(x)
        #for i in range(len(x_hf)):
        #   x_hfl.append(torch.zeros(x_hf[i].size()).cuda())
        x_midf.append(torch.zeros(x_hf[0].size()).cuda())
        #x_hfl.append(x_hf[0])
        x_midf.append(x_hf[1])
        x_midf.append(x_hf[2])
        x_midf.append(x_hf[3])
        x_midf.append(x_hf[4])
        x_midf.append(x_hf[5])

        x_hfl.append(x_hf[0])
        x_hfl.append(torch.zeros(x_hf[1].size()).cuda())
        x_hfl.append(torch.zeros(x_hf[2].size()).cuda())
        x_hfl.append(torch.zeros(x_hf[3].size()).cuda())
        x_hfl.append(torch.zeros(x_hf[4].size()).cuda())
        x_hfl.append(torch.zeros(x_hf[5].size()).cuda())

        x_lfl = torch.zeros(x_lf.size()).cuda()
        x_mid = self.ixfm((x_lfl,x_midf))
        #x_hf = self.ixfm((x_lfl,x_hf))
        #x_lf = self.ixfm((x_lf,x_hfl))
        #x_mid = self.ixfm((x_lfl,x_hfl))
        x_hlf = self.ixfm((x_lf,x_hfl))
        x_hlf = F.interpolate(x_hlf, size=(192,192))
        ##################################################################
           # ResNet
        x = self.layer0[0](x)
        #print(x.shape)
        x = self.layer0[1](x)
        #print(x.shape)
        x = self.layer0[2](x)
        #print(x.shape)
        x = self.layer0[3](x)
        xp = x
        # ########################### DWT ##################################
        # x_lf,x_hf = self.xfm(x)
        # for i in range(len(x_hf)):
        #     x_hfl.append(torch.zeros(x_hf[i].size()).cuda())
        # #x_hfl.append(torch.zeros(x_hf[0].size()).cuda())
        # #x_hfl.append(x_hf[1])
        # #x_hfl.append(x_hf[2])
        # #x_lfl = torch.zeros(x_lf.size()).cuda()
        # #x_hf = self.ixfm((x_lfl,x_hfl))
        # x_lf = self.ixfm((x_lf,x_hfl))
        # #x_mid = self.ixfm((x_lfl,x_hfl))
        # ##################################################################
        if(training==True and p<0.5):
           x = self.Normalization_Perturbation_Plus_mid(xp)
        # OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(x),scale_factor =(1.205,1.205)))) #layersize256 #output320
        # #if(training==True and p<0.5):
        # #   OCout = self.Normalization_Perturbation_Plus(OCout)
        # OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))#layersize320 #output400
        # #if(training==True and p<0.5):
        # #   OCout = self.Normalization_Perturbation_Plus(OCout)
        # OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.2,1.2))))#layersize400 output500
        # #print(OCout.shape)
        # OCout = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), size =(int(h/2),int(w/2)))))#layersize500 output625
        
        OCout = F.relu(self.OC1_bn(self.OClayer1(xp)))
        if(training==True and p<0.5):
            OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC2_bn(self.OClayer2(OCout)))
        if(training==True and p<0.5):
            OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC3_bn(self.OClayer3(OCout)))
        OCout = F.relu(self.OC4_bn(self.OClayer4(OCout)))
        
        x_tuple = self.layer1([x, w_arr])  # 400
        if(training==True and p<0.5):
           x_tuple[0] = self.Normalization_Perturbation_Plus_mid(x_tuple[0])
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
        #dec2 = self.final2(dec1)
        #print(dec1.shape)
        #print(hey)
        #dec1 = Upsample(dec1, (int(h/2),int(w/2)))

        ################# FU in GMU ##############
        '''batch, _,_,_ = dec1.shape
        dec1_fft = torch.view_as_real(torch.fft.rfft2(dec1))
        dec1_fft = dec1_fft.permute(0, 1, 4, 2, 3).contiguous()
        dec1_fft = dec1_fft.view((batch, -1,) + dec1_fft.size()[3:])

        OCout_fft = torch.view_as_real(torch.fft.rfft2(OCout))
        OCout_fft = OCout_fft.permute(0, 1, 4, 2, 3).contiguous()
        OCout_fft = OCout_fft.view((batch, -1,) + OCout_fft.size()[3:])

        hv = self.relu(self.hv_bn(self.hv(dec1_fft)))
        ht = self.relu(self.ht_bn(self.ht(OCout_fft)))
        hv = hv.view((batch, -1, 2,) + hv.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        ht = ht.view((batch, -1, 2,) + ht.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        hv = torch.view_as_complex(hv)
        ht = torch.view_as_complex(ht)
        hv = torch.fft.irfft2(hv)
        ht = torch.fft.irfft2(ht)'''
        hv = self.tanh(self.hv(dec1))
        ht = self.tanh(self.ht(OCout))
        z = self.sigmoid(self.z(torch.cat([dec1,OCout],dim=1)))
        #hv = F.interpolate(hv, size=(z.shape[2],z.shape[3]))
        #ht = F.interpolate(ht, size=(z.shape[2],z.shape[3]))
        dec1 = z*hv + (1-z)*ht

        ####################################################
        #print(dec2.shape)
        dec2 = self.final2(dec1)
        #dec2 = torch.add(OCout, dec2)
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
        

class KiLabMAO(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,0,0,0,0,0], use_wtloss=False):
        super(KiLabMAO, self).__init__()
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
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))
        
        # self.dsn = nn.Sequential(
        #     nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
        #     Norm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(0.1),
        #     nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # )

        self.OCenclayer1 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)
        self.OC1_enc_bn = nn.BatchNorm2d(64)
        self.OCenclayer2 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)
        self.OC2_enc_bn = nn.BatchNorm2d(64)
        self.OCenclayer3 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC3_enc_bn = nn.BatchNorm2d(128)

        self.OCdeclayer1 = nn.Conv2d(128,64,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC1_dec_bn = nn.BatchNorm2d(64)
        self.OCdeclayer2 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC2_dec_bn = nn.BatchNorm2d(64)
        self.OCdeclayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC3_dec_bn = nn.BatchNorm2d(64)



        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hv = nn.Conv2d(64,64,kernel_size=1,stride=1,padding=1)

        self.ht = nn.Conv2d(64,64,kernel_size=1,stride=1,padding=1)

        self.z = nn.Conv2d(128,64,kernel_size=1,stride=1,padding=1)

        #initialize_weights(self.dsn)
        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        self.three_input_layer = False

        #self.xfm = DWTForward(J=6, mode='zero', wave='db3')
        #self.ixfm = DWTInverse(mode='zero', wave='db3')

    def Normalization_Perturbation(self, feat):
    # feat: input features of size (B, C, H, W)
        feat_mean = feat.mean((2, 3), keepdim=True) # size: B, C, 1, 1
        ones_mat = torch.ones_like(feat_mean)
        alpha = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
        beta = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
        output = alpha * feat - alpha * feat_mean + beta * feat_mean
        return output # size: B, C, H, W
    
    def Normalization_Perturbation_Plus_mid(self, feat):
        feat_mean = feat.mean((2, 3), keepdim=True)
        ones_mat = torch.ones_like(feat_mean)
        zeros_mat = torch.zeros_like(feat_mean)
        mean_diff = torch.std(feat_mean, 0, keepdim=True)
        mean_scale = mean_diff / mean_diff.max() * 1.5
        alpha = torch.normal(ones_mat, 0.5 * ones_mat)
        beta = 1 + torch.normal(zeros_mat, 0.5 * ones_mat) * mean_scale
        output = alpha * feat - alpha * feat_mean + beta * feat_mean
        return output
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
        w_arr = []
        x_midf = []
        x_hfl = []
        x_size = x.size()  # 800
        h,w = x_size[2:]
        #print(x_size)
        #print(x.shape)
        # ########################### DWT ##################################
        # x_lf,x_hf = self.xfm(x)
        # #for i in range(len(x_hf)):
        # #   x_hfl.append(torch.zeros(x_hf[i].size()).cuda())
        # x_midf.append(torch.zeros(x_hf[0].size()).cuda())
        # #x_hfl.append(x_hf[0])
        # x_midf.append(x_hf[1])
        # x_midf.append(x_hf[2])
        # x_midf.append(x_hf[3])
        # x_midf.append(x_hf[4])
        # x_midf.append(x_hf[5])

        # x_hfl.append(x_hf[0])
        # x_hfl.append(torch.zeros(x_hf[1].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[2].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[3].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[4].size()).cuda())
        # x_hfl.append(torch.zeros(x_hf[5].size()).cuda())

        # x_lfl = torch.zeros(x_lf.size()).cuda()
        # x_mid = self.ixfm((x_lfl,x_midf))
        # #x_hf = self.ixfm((x_lfl,x_hf))
        # #x_lf = self.ixfm((x_lf,x_hfl))
        # #x_mid = self.ixfm((x_lfl,x_hfl))
        # x_hlf = self.ixfm((x_lf,x_hfl))
        # x_hlf = F.interpolate(x_hlf, size=(192,192))
        # ##################################################################
           # ResNet
        x = self.layer0[0](x)
        #print(x.shape)
        x = self.layer0[1](x)
        #print(x.shape)
        x = self.layer0[2](x)
        #print(x.shape)
        x = self.layer0[3](x)
        #xp = x
        # ########################### DWT ##################################
        # x_lf,x_hf = self.xfm(x)
        # for i in range(len(x_hf)):
        #     x_hfl.append(torch.zeros(x_hf[i].size()).cuda())
        # #x_hfl.append(torch.zeros(x_hf[0].size()).cuda())
        # #x_hfl.append(x_hf[1])
        # #x_hfl.append(x_hf[2])
        # #x_lfl = torch.zeros(x_lf.size()).cuda()
        # #x_hf = self.ixfm((x_lfl,x_hfl))
        # x_lf = self.ixfm((x_lf,x_hfl))
        # #x_mid = self.ixfm((x_lfl,x_hfl))
        # ##################################################################
        #if(training==True and p<0.5):
        #   x = self.Normalization_Perturbation_Plus_mid(xp)
        xp = x
        _,_,oh,ow = xp.size()
        OCout = F.relu(self.OC1_enc_bn(self.OCenclayer1(xp))) #layersize256 #output320
        #if(training==True and p<0.5):
        #   OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC2_enc_bn(F.interpolate(self.OCenclayer2(OCout), scale_factor =(1.17,1.17))))#layersize320 #output400
        #if(training==True and p<0.5):
        #   OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC3_enc_bn(F.interpolate(self.OCenclayer3(OCout), scale_factor =(1.145,1.145))))#layersize400 output500
        #print(OCout.shape)
        OCout = F.relu(self.OC1_dec_bn(F.interpolate(self.OCdeclayer1(OCout), size =(0.875,0.875))))#layersize500 output625
        OCout = F.relu(self.OC2_dec_bn(F.interpolate(self.OCdeclayer2(OCout), size =(0.858,0.858))))#layersize500 output625
        OCout = F.relu(self.OC3_dec_bn(F.interpolate(self.OCdeclayer3(OCout), size =(int(oh),int(ow)))))#layersize500 output625
        
        
        x_tuple = self.layer1([x, w_arr])  # 400
        #if(training==True and p<0.5):
        #   x_tuple[0] = self.Normalization_Perturbation_Plus_mid(x_tuple[0])
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
        #dec2 = self.final2(dec1)
        #print(dec1.shape)
        #print(hey)
        #dec1 = Upsample(dec1, (int(h/2),int(w/2)))

        ################# FU in GMU ##############
        '''batch, _,_,_ = dec1.shape
        dec1_fft = torch.view_as_real(torch.fft.rfft2(dec1))
        dec1_fft = dec1_fft.permute(0, 1, 4, 2, 3).contiguous()
        dec1_fft = dec1_fft.view((batch, -1,) + dec1_fft.size()[3:])

        OCout_fft = torch.view_as_real(torch.fft.rfft2(OCout))
        OCout_fft = OCout_fft.permute(0, 1, 4, 2, 3).contiguous()
        OCout_fft = OCout_fft.view((batch, -1,) + OCout_fft.size()[3:])

        hv = self.relu(self.hv_bn(self.hv(dec1_fft)))
        ht = self.relu(self.ht_bn(self.ht(OCout_fft)))
        hv = hv.view((batch, -1, 2,) + hv.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        ht = ht.view((batch, -1, 2,) + ht.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        hv = torch.view_as_complex(hv)
        ht = torch.view_as_complex(ht)
        hv = torch.fft.irfft2(hv)
        ht = torch.fft.irfft2(ht)'''
        hv = self.tanh(self.hv(dec1))
        ht = self.tanh(self.ht(OCout))
        z = self.sigmoid(self.z(torch.cat([dec1,OCout],dim=1)))
        #hv = F.interpolate(hv, size=(z.shape[2],z.shape[3]))
        #ht = F.interpolate(ht, size=(z.shape[2],z.shape[3]))
        dec1 = z*hv + (1-z)*ht

        ####################################################
        #print(dec2.shape)
        dec2 = self.final2(dec1)
        #dec2 = torch.add(OCout, dec2)
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
        
        # self.dsn = nn.Sequential(
        #     nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
        #     Norm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(0.1),
        #     nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # )
        # initialize_weights(self.dsn)


        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        self.three_input_layer = False

        #self.xfm = DWTForward(J=4, mode='zero', wave='db3')
        #self.ixfm = DWTInverse(mode='zero', wave='db3')

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
        ''''########################### DWT ##################################
        x_lf,x_hf = self.xfm(x)
        #for i in range(len(x_hf)):
        #    x_hfl.append(torch.zeros(x_hf[i].size()).cuda())
        x_hfl.append(torch.zeros(x_hf[0].size()).cuda())
        x_hfl.append(x_hf[1])
        x_hfl.append(x_hf[2])
        x_hfl.append(x_hf[3])
        #x_hfl.append(x_hf[4])
        #x_hfl.append(x_hf[5])
        x_lfl = torch.zeros(x_lf.size()).cuda()
        #x_hf = self.ixfm((x_lfl,x_hfl))
        #x_lf = self.ixfm((x_lf,x_hfl))
        x_mid = self.ixfm((x_lfl,x_hfl))
        ##################################################################'''
        # ResNet
        x = self.layer0[0](x)
        #print(x.shape)
        x = self.layer0[1](x)
        #print(x.shape)
        x = self.layer0[2](x)
        #print(x.shape)
        x = self.layer0[3](x)
        
        #if(training==True and p<0.5):
        #    x = self.Normalization_Perturbation_Plus(x)

        x_tuple = self.layer1([x, w_arr])  # 400
        #f_map = x_tuple[0]

        #if(training==True and p<0.5):
        #    x_tuple[0] = self.Normalization_Perturbation_Plus(x_tuple[0])

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
        
class LabMAO_MAOEncPerturbbridge_noskipconn_randomOC_godmodel(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,0,0,0,0,0], use_wtloss=False):
        super(LabMAO_MAOEncPerturbbridge_noskipconn_randomOC_godmodel, self).__init__()
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

        self.OCdeclayer1 = nn.Conv2d(256,128,kernel_size=3, stride=1, padding=1).requires_grad_(False) #bridge
        self.OC1_decbn = nn.BatchNorm2d(128).requires_grad_(False)
        self.OCdeclayer2 = nn.Conv2d(128,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC2_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC3_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer4 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC4_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer5 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC5_decbn = nn.BatchNorm2d(64).requires_grad_(False)

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
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer5)
        initialize_weights_kaimingnormal_forOC(self.OC5_decbn)
    
        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        self.three_input_layer = False

    # def Normalization_Perturbation(self, feat):
    # # feat: input features of size (B, C, H, W)
    #     feat_mean = feat.mean((2, 3), keepdim=True) # size: B, C, 1, 1
    #     ones_mat = torch.ones_like(feat_mean)
    #     alpha = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
    #     beta = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
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
        w_arr = []
        x_size = x.size()  # 800
        h,w = x_size[2:]

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
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer5)
            initialize_weights_kaimingnormal_forOC(self.OC5_decbn)

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
        if(training==True and p<0.5):
            
            OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp),scale_factor =(1.205,1.205))))
            OC1shape = OCout.shape[2], OCout.shape[3]
            
            OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))
            OC2shape = OCout.shape[2], OCout.shape[3]

            # if(training==True and p<0.5):
            #     OCout = self.Normalization_Perturbation_Plus(OCout)
            OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.2,1.2))))
            OC3shape = OCout.shape[2], OCout.shape[3]

            OCout = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), size =(int(h/2),int(w/2)))))

            OCout = F.relu(self.OC1_decbn(F.interpolate(self.OCdeclayer1(OCout), size =(int(h/2),int(w/2)))))
            # print("after dec1: ", OCout.shape)
            OCout = F.relu(self.OC2_decbn(F.interpolate(self.OCdeclayer2(OCout), size =(OC3shape))))
            # print("after dec2: ", OCout.shape)
            OCout = F.relu(self.OC3_decbn(F.interpolate(self.OCdeclayer3(OCout), size =(OC2shape))))
            # print("after dec3", OCout.shape)
            OCout = F.relu(self.OC4_decbn(F.interpolate(self.OCdeclayer4(OCout), size =(OC1shape))))
            # print("after dec4", OCout.shape)
            OCout = F.relu(self.OC5_decbn(F.interpolate(self.OCdeclayer5(OCout), size =((math.ceil(h/4),math.ceil(w/4))))))

        ##GMU fusion##
        # hv = self.tanh(self.hv(x))
        # ht = self.tanh(self.ht(OCout))
        # z = self.sigmoid(self.z(torch.cat([x,OCout],dim=1)))
        # x = z*hv + (1-z)*ht
        if(training==True and p<0.5):
            x = torch.add(OCout, x)
        ##################
        x_tuple = self.layer1([x, w_arr])  # 400
        #f_map = x_tuple[0]
        if(training==True and p2<0.5):
            x_tuple[0] = self.Normalization_Perturbation_Plus(x_tuple[0])
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
        #dec1 = Upsample(dec1, (int(h/2),int(w/2)))


        
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
        

class LabMAO_MAOEnc_noskipconn_learnable(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,0,0,0,0,0], use_wtloss=False):
        super(LabMAO_MAOEnc_noskipconn_learnable, self).__init__()
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

        self.OClayer1 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)
        self.OC1_bn = nn.BatchNorm2d(64)
        self.OClayer2 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)
        self.OC2_bn = nn.BatchNorm2d(64)
        self.OClayer3 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC3_bn = nn.BatchNorm2d(128)
        self.OClayer4 = nn.Conv2d(128,256,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC4_bn = nn.BatchNorm2d(256)

        self.OCdeclayer1 = nn.Conv2d(256,128,kernel_size=3, stride=1, padding=1)
        self.OC1_decbn = nn.BatchNorm2d(128)
        self.OCdeclayer2 = nn.Conv2d(128,64,kernel_size=3, stride=1, padding=1)
        self.OC2_decbn = nn.BatchNorm2d(64)
        self.OCdeclayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC3_decbn = nn.BatchNorm2d(64)
        self.OCdeclayer4 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC4_decbn = nn.BatchNorm2d(64)

        # initialize_weights_kaimingnormal_forOC(self.OClayer1)
        # initialize_weights_kaimingnormal_forOC(self.OC1_bn)
        # initialize_weights_kaimingnormal_forOC(self.OClayer2)
        # initialize_weights_kaimingnormal_forOC(self.OC2_bn)
        # initialize_weights_kaimingnormal_forOC(self.OClayer3)
        # initialize_weights_kaimingnormal_forOC(self.OC3_bn)
        # initialize_weights_kaimingnormal_forOC(self.OClayer4)
        # initialize_weights_kaimingnormal_forOC(self.OC4_bn)
        # initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
        # initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
        # initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
        # initialize_weights_kaimingnormal_forOC(self.OC2_decbn)
        # initialize_weights_kaimingnormal_forOC(self.OCdeclayer3)
        # initialize_weights_kaimingnormal_forOC(self.OC3_decbn)
        # initialize_weights_kaimingnormal_forOC(self.OCdeclayer4)
        # initialize_weights_kaimingnormal_forOC(self.OC4_decbn)
    
        # initialize_weights(self.aspp)
        # initialize_weights(self.bot_aspp)
        # initialize_weights(self.bot_fine)
        # initialize_weights(self.final1)
        # initialize_weights(self.final2)

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
        # p3 = random.random()
        w_arr = []
        x_hfi_bands = []
        x_size = x.size()  # 800
        h,w = x_size[2:]
        b,_,_,_ = x.shape

        # if(training==True and p3<0.5):
        #     i = random.randint(2,b-1)
        #     ########################### DWT ##################################
        #     # x_lf,x_hf = self.xfm(torch.unsqueeze(x[i],0))
        #     # NP_img = self.Normalization_Perturbation(torch.unsqueeze(x[i],0))
        #     # x_NP_lf, x_NP_hf = self.xfm(NP_img)
        #     # x_hfi_bands.append(x_NP_hf[0][:,:,0,:,:])
        #     # x_hfi_bands.append(x_hf[0][:,:,1,:,:])
        #     # x_hfi_bands.append(x_hf[0][:,:,2,:,:])
        #     # tensors=(x_hfi_bands[0], x_hfi_bands[1],x_hfi_bands[2])
        #     # x_hf[0] = torch.stack(tensors,2)
        #     # x_i = self.ixfm((x_lf,x_hf))
        #     # x = torch.cat([x[0:i], x[i+1:]])
        #     # x = torch.cat([x,x_i],dim=0)
        #     # #for i in range(len(x_hf)):
        #     # #    x_hfl.append(torch.zeros(x_hf[i].size()).cuda())
        #     # # x_hfl.append(torch.zeros(x_hf[0].size()).cuda())
        #     # # x_hfl.append(x_hf[1])
        #     # # x_hfl.append(x_hf[2])
        #     # # x_hfl.append(x_hf[3])
        #     # # x_hfl.append(x_hf[4])
        #     # # x_hfl.append(x_hf[5])
        #     # #x_lfl = torch.zeros(x_lf.size()).cuda()
        #     # #x_hf = self.ixfm((x_lfl,x_hf))
        #     # #x_lf = self.ixfm((x_lf,x_hfl))
        #     # #x_mid = self.ixfm((x_lfl,x_hf))
        #     IN_img = self.IN(torch.unsqueeze(x[i],0))
        #     x = torch.cat([x[0:i], x[i+1:]])
        #     x = torch.cat([x,IN_img],dim=0)

        #     ##################################################################


        # if(training==True and p<0.5):
        #     initialize_weights_kaimingnormal_forOC(self.OClayer1)
        #     initialize_weights_kaimingnormal_forOC(self.OC1_bn)
        #     initialize_weights_kaimingnormal_forOC(self.OClayer2)
        #     initialize_weights_kaimingnormal_forOC(self.OC2_bn)
        #     initialize_weights_kaimingnormal_forOC(self.OClayer3)
        #     initialize_weights_kaimingnormal_forOC(self.OC3_bn)
        #     initialize_weights_kaimingnormal_forOC(self.OClayer4)
        #     initialize_weights_kaimingnormal_forOC(self.OC4_bn)
        #     initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
        #     initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
        #     initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
        #     initialize_weights_kaimingnormal_forOC(self.OC2_decbn)
        #     initialize_weights_kaimingnormal_forOC(self.OCdeclayer3)
        #     initialize_weights_kaimingnormal_forOC(self.OC3_decbn)
        #     initialize_weights_kaimingnormal_forOC(self.OCdeclayer4)
        #     initialize_weights_kaimingnormal_forOC(self.OC4_decbn)

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
            
        OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp),scale_factor =(1.205,1.205))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.2,1.2))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout_dec = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), size =(int(h/2),int(w/2)))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)

        OCout = F.relu(self.OC1_decbn(F.interpolate(self.OCdeclayer1(OCout_dec), size =(int(h/2),int(w/2)))))
        OCout = F.relu(self.OC2_decbn(F.interpolate(self.OCdeclayer2(OCout), scale_factor =(0.838,0.838))))
        OCout = F.relu(self.OC3_decbn(F.interpolate(self.OCdeclayer3(OCout), scale_factor =(0.798,0.798))))
        OCout = F.relu(self.OC4_decbn(F.interpolate(self.OCdeclayer4(OCout), size =(math.ceil(h/4),math.ceil(w/4)))))

        ##GMU fusion##
        # hv = self.tanh(self.hv(x))
        # ht = self.tanh(self.ht(OCout))
        # z = self.sigmoid(self.z(torch.cat([x,OCout],dim=1)))
        # x = z*hv + (1-z)*ht
        #if(training==True and p<0.5):
        x = torch.add(OCout, x)
        ##################
        x_tuple = self.layer1([x, w_arr])  # 400
        #f_map = x_tuple[0]
        if(training==True and p2<0.5):
            x_tuple[0] = self.Normalization_Perturbation_Plus(x_tuple[0])
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
        # if(training==True and p3<0.3):
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

class rebuttal_MRFPPlus_10layer(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,4,4,4,0,0], use_wtloss=False):
        super(rebuttal_MRFPPlus_10layer, self).__init__()
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
        self.OClayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC3_bn = nn.BatchNorm2d(64).requires_grad_(False)
        #self.OC3_IN = nn.InstanceNorm2d(128,affine=False).requires_grad_(False)
        self.OClayer4 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC4_bn = nn.BatchNorm2d(128).requires_grad_(False)
        self.OClayer5 = nn.Conv2d(128,256,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC5_bn = nn.BatchNorm2d(256).requires_grad_(False)

        self.OCdeclayer1 = nn.Conv2d(256,128,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC1_decbn = nn.BatchNorm2d(128).requires_grad_(False)
        self.OCdeclayer2 = nn.Conv2d(128,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC2_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC3_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer4 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC4_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer5 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC5_decbn = nn.BatchNorm2d(64).requires_grad_(False)

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
        initialize_weights_kaimingnormal_forOC(self.OClayer5)
        initialize_weights_kaimingnormal_forOC(self.OC5_bn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
        initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
        initialize_weights_kaimingnormal_forOC(self.OC2_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer3)
        initialize_weights_kaimingnormal_forOC(self.OC3_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer4)
        initialize_weights_kaimingnormal_forOC(self.OC4_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer5)
        initialize_weights_kaimingnormal_forOC(self.OC5_decbn)
    
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
            initialize_weights_kaimingnormal_forOC(self.OClayer5)
            initialize_weights_kaimingnormal_forOC(self.OC5_bn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
            initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
            initialize_weights_kaimingnormal_forOC(self.OC2_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer3)
            initialize_weights_kaimingnormal_forOC(self.OC3_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer4)
            initialize_weights_kaimingnormal_forOC(self.OC4_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer5)
            initialize_weights_kaimingnormal_forOC(self.OC5_decbn)

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
            
        OCout1 = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp),scale_factor =(1.15,1.15))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        # OCout2 = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout1), scale_factor =(1.2,1.2))))
        OCout2 = F.relu(F.interpolate(self.OClayer2(OCout1), scale_factor =(1.15,1.15)))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout3 = F.relu(F.interpolate(self.OClayer3(OCout2), scale_factor =(1.15,1.15)))
        OCout4 = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout3), scale_factor =(1.15,1.15))))
        # OCout3 = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout2), scale_factor =(1.2,1.2))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout_dec = F.relu(self.OC5_bn(F.interpolate(self.OClayer5(OCout4), size =(int(h/2),int(w/2)))))
        # print(OCout_dec.shape)
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)

        OCoutdeclayer1 = F.relu(self.OC1_decbn(F.interpolate(self.OCdeclayer1(OCout_dec), size =(int(h/2),int(w/2)))))
        OCoutdeclayer2 = F.relu(self.OC2_decbn(F.interpolate(self.OCdeclayer2(OCoutdeclayer1), scale_factor =(0.85,0.85))))
        OCoutdeclayer3 = F.relu(self.OC3_decbn(F.interpolate(self.OCdeclayer3(OCoutdeclayer2), scale_factor =(0.85,0.85))))
        OCoutdeclayer4 = F.relu(self.OC4_decbn(F.interpolate(self.OCdeclayer4(OCoutdeclayer3), scale_factor =(0.85,0.85))))
        OCout = F.relu(self.OC5_decbn(F.interpolate(self.OCdeclayer5(OCoutdeclayer4), size =(math.ceil(h/4),math.ceil(w/4)))))

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
        
class rebuttal_MRFPPlus_12layer(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,4,4,4,0,0], use_wtloss=False):
        super(rebuttal_MRFPPlus_12layer, self).__init__()
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
        self.OClayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC3_bn = nn.BatchNorm2d(64).requires_grad_(False)
        #self.OC3_IN = nn.InstanceNorm2d(128,affine=False).requires_grad_(False)
        self.OClayer4 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC4_bn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OClayer5 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC5_bn = nn.BatchNorm2d(128).requires_grad_(False)
        self.OClayer6 = nn.Conv2d(128,256,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC6_bn = nn.BatchNorm2d(256).requires_grad_(False)

        self.OCdeclayer1 = nn.Conv2d(256,128,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC1_decbn = nn.BatchNorm2d(128).requires_grad_(False)
        self.OCdeclayer2 = nn.Conv2d(128,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC2_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC3_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer4 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC4_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer5 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC5_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer6 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC6_decbn = nn.BatchNorm2d(64).requires_grad_(False)

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
        initialize_weights_kaimingnormal_forOC(self.OClayer5)
        initialize_weights_kaimingnormal_forOC(self.OC5_bn)
        initialize_weights_kaimingnormal_forOC(self.OClayer6)
        initialize_weights_kaimingnormal_forOC(self.OC6_bn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
        initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
        initialize_weights_kaimingnormal_forOC(self.OC2_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer3)
        initialize_weights_kaimingnormal_forOC(self.OC3_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer4)
        initialize_weights_kaimingnormal_forOC(self.OC4_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer5)
        initialize_weights_kaimingnormal_forOC(self.OC5_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer6)
        initialize_weights_kaimingnormal_forOC(self.OC6_decbn)
    
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
            initialize_weights_kaimingnormal_forOC(self.OClayer5)
            initialize_weights_kaimingnormal_forOC(self.OC5_bn)
            initialize_weights_kaimingnormal_forOC(self.OClayer6)
            initialize_weights_kaimingnormal_forOC(self.OC6_bn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
            initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
            initialize_weights_kaimingnormal_forOC(self.OC2_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer3)
            initialize_weights_kaimingnormal_forOC(self.OC3_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer4)
            initialize_weights_kaimingnormal_forOC(self.OC4_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer5)
            initialize_weights_kaimingnormal_forOC(self.OC5_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer6)
            initialize_weights_kaimingnormal_forOC(self.OC6_decbn)

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
            
        OCout1 = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp),scale_factor =(1.12,1.12))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        # OCout2 = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout1), scale_factor =(1.2,1.2))))
        OCout2 = F.relu(F.interpolate(self.OClayer2(OCout1), scale_factor =(1.12,1.12)))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout3 = F.relu(F.interpolate(self.OClayer3(OCout2), scale_factor =(1.12,1.12)))
        OCout4 = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout3), scale_factor =(1.12,1.12))))
        OCout5 = F.relu(self.OC5_bn(F.interpolate(self.OClayer5(OCout4), scale_factor =(1.12,1.12))))
        # OCout3 = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout2), scale_factor =(1.2,1.2))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout_dec = F.relu(self.OC6_bn(F.interpolate(self.OClayer6(OCout5), size =(int(h/2),int(w/2)))))
        # print(OCout_dec.shape)
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)

        OCoutdeclayer1 = F.relu(self.OC1_decbn(F.interpolate(self.OCdeclayer1(OCout_dec), size =(int(h/2),int(w/2)))))
        OCoutdeclayer2 = F.relu(self.OC2_decbn(F.interpolate(self.OCdeclayer2(OCoutdeclayer1), scale_factor =(0.88,0.88)))) 
        OCoutdeclayer3 = F.relu(self.OC3_decbn(F.interpolate(self.OCdeclayer3(OCoutdeclayer2), scale_factor =(0.88,0.88)))) 
        OCoutdeclayer4 = F.relu(self.OC4_decbn(F.interpolate(self.OCdeclayer4(OCoutdeclayer3), scale_factor =(0.88,0.88)))) 
        OCoutdeclayer5 = F.relu(self.OC5_decbn(F.interpolate(self.OCdeclayer5(OCoutdeclayer4), scale_factor =(0.88,0.88)))) 
        OCout = F.relu(self.OC6_decbn(F.interpolate(self.OCdeclayer6(OCoutdeclayer5), size =(math.ceil(h/4),math.ceil(w/4)))))

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

class rebuttal_MRFPPlus_6layer(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,4,4,4,0,0], use_wtloss=False):
        super(rebuttal_MRFPPlus_6layer, self).__init__()
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


        #self.OC3_IN = nn.InstanceNorm2d(128,affine=False).requires_grad_(False)
        self.OClayer1 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC1_bn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OClayer2 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC2_bn = nn.BatchNorm2d(128).requires_grad_(False)
        self.OClayer3 = nn.Conv2d(128,256,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC3_bn = nn.BatchNorm2d(256).requires_grad_(False)

        self.OCdeclayer1 = nn.Conv2d(256,128,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC1_decbn = nn.BatchNorm2d(128).requires_grad_(False)
        self.OCdeclayer2 = nn.Conv2d(128,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC2_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC3_decbn = nn.BatchNorm2d(64).requires_grad_(False)


        initialize_weights_kaimingnormal_forOC(self.OClayer1)
        initialize_weights_kaimingnormal_forOC(self.OC1_bn)
        #initialize_weights_kaimingnormal_forOC(self.OC1_IN)
        initialize_weights_kaimingnormal_forOC(self.OClayer2)
        initialize_weights_kaimingnormal_forOC(self.OC2_bn)
        #initialize_weights_kaimingnormal_forOC(self.OC2_IN)
        initialize_weights_kaimingnormal_forOC(self.OClayer3)
        initialize_weights_kaimingnormal_forOC(self.OC3_bn)
        #initialize_weights_kaimingnormal_forOC(self.OC3_IN)

        initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
        initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
        initialize_weights_kaimingnormal_forOC(self.OC2_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer3)
        initialize_weights_kaimingnormal_forOC(self.OC3_decbn)

    
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

            initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
            initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
            initialize_weights_kaimingnormal_forOC(self.OC2_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer3)
            initialize_weights_kaimingnormal_forOC(self.OC3_decbn)

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
            
        OCout1 = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp),scale_factor =(1.26,1.26))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        # OCout2 = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout1), scale_factor =(1.2,1.2))))
        OCout2 = F.relu(F.interpolate(self.OClayer2(OCout1), scale_factor =(1.26,1.26)))
        # if(training==True and p<0.5):
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout_dec = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout2), size =(int(h/2),int(w/2)))))
        # print(OCout_dec.shape)
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)

        OCoutdeclayer1 = F.relu(self.OC1_decbn(F.interpolate(self.OCdeclayer1(OCout_dec), size =(int(h/2),int(w/2)))))
        OCoutdeclayer2 = F.relu(self.OC2_decbn(F.interpolate(self.OCdeclayer2(OCoutdeclayer1), scale_factor =(0.72,0.72)))) 

        OCout = F.relu(self.OC3_decbn(F.interpolate(self.OCdeclayer3(OCoutdeclayer2), size =(math.ceil(h/4),math.ceil(w/4)))))

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
        
class rebuttal_MRFPPlus_4layer(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,4,4,4,0,0], use_wtloss=False):
        super(rebuttal_MRFPPlus_4layer, self).__init__()
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


        #self.OC3_IN = nn.InstanceNorm2d(128,affine=False).requires_grad_(False)

        self.OClayer1 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC1_bn = nn.BatchNorm2d(128).requires_grad_(False)
        self.OClayer2 = nn.Conv2d(128,256,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC2_bn = nn.BatchNorm2d(256).requires_grad_(False)

        self.OCdeclayer1 = nn.Conv2d(256,128,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC1_decbn = nn.BatchNorm2d(128).requires_grad_(False)
        self.OCdeclayer2 = nn.Conv2d(128,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC2_decbn = nn.BatchNorm2d(64).requires_grad_(False)


        initialize_weights_kaimingnormal_forOC(self.OClayer1)
        initialize_weights_kaimingnormal_forOC(self.OC1_bn)
        initialize_weights_kaimingnormal_forOC(self.OClayer2)
        initialize_weights_kaimingnormal_forOC(self.OC2_bn)


        initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
        initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
        initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
        initialize_weights_kaimingnormal_forOC(self.OC2_decbn)


    
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
            initialize_weights_kaimingnormal_forOC(self.OClayer2)
            initialize_weights_kaimingnormal_forOC(self.OC2_bn)


            initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
            initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
            initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
            initialize_weights_kaimingnormal_forOC(self.OC2_decbn)

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
            
        OCout1 = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp),scale_factor =(1.4,1.4))))

        OCout_dec = F.relu(F.interpolate(self.OClayer2(OCout1), size =(int(h/2),int(w/2))))


        OCoutdeclayer1 = F.relu(self.OC1_decbn(F.interpolate(self.OCdeclayer1(OCout_dec), size =(int(h/2),int(w/2)))))
        OCout = F.relu(self.OC2_decbn(F.interpolate(self.OCdeclayer2(OCoutdeclayer1), size =(math.ceil(h/4),math.ceil(w/4))))) 


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

        if training:
            loss1 = self.criterion(main_out, gts)
            return_loss = loss1
            return return_loss
        else:
            return main_out

class hrfp_noplus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,4,4,4,0,0], use_wtloss=False):
        super(hrfp_noplus, self).__init__()
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

        # self.OCdec_inter = nn.Conv2d(64,256,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        # self.OCenc_inter = nn.Conv2d(128,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)


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

        # initialize_weights_kaimingnormal_forOC(self.OCdec_inter)
        # initialize_weights_kaimingnormal_forOC(self.OCenc_inter)
    
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
    
    # def Normalization_Perturbation_Plus(self, feat):
    #     feat_mean = feat.mean((2, 3), keepdim=True)
    #     ones_mat = torch.ones_like(feat_mean)
    #     zeros_mat = torch.zeros_like(feat_mean)
    #     mean_diff = torch.std(feat_mean, 0, keepdim=True)
    #     mean_scale = mean_diff / mean_diff.max() * 1.5
    #     alpha = torch.normal(ones_mat, 0.75 * ones_mat)
    #     beta = 1 + torch.normal(zeros_mat, 0.75 * ones_mat) * mean_scale
    #     output = alpha * feat - alpha * feat_mean + beta * feat_mean
    #     return output
    
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
        # if(training==True and p3<0.5):
        #     i = random.randint(2,b-1)
        #     ########################### APR ##################################
        #     NP_img = self.Normalization_Perturbation(torch.unsqueeze(x[i],0))
        #     # fft_1 = torch.fft.fftshift(torch.fft.rfftn(NP_img))

        #     # abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)
        #     # abs2 = abs_1 * 0.5

        #     # fft_2 = abs2*torch.exp((1j) * angle_1)

        #     # x_final2 = torch.fft.irfftn(torch.fft.ifftshift(fft_2))
        #     # x_final2[x_final2>255]=255
        #     # x_final2[x_final2<0]=0
            
        #     x = torch.cat([x[0:i], x[i+1:]])
        #     x = torch.cat([x,NP_img],dim=0)


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
        # initialize_weights_kaimingnormal_forOC(self.OCdec_inter)
        # initialize_weights_kaimingnormal_forOC(self.OCenc_inter)

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
        # if(training==True and p2<0.5):
            # x = self.Normalization_Perturbation_Plus(xp)
            # print(x)
            # print(hey)
        if(training==True and p<0.5):
            OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp),scale_factor =(1.205,1.205))))
            # print(OCout.shape)
            # if(training==True and p<0.5):
            #     OCout = self.Normalization_Perturbation_Plus(OCout)
            OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))
            # print(OCout.shape)
            # if(training==True and p<0.5):
            #     OCout = self.Normalization_Perturbation_Plus(OCout)
            OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.2,1.2))))
            # print(OCout.shape)
            # if(training==True and p<0.5):
            #     OCout = self.Normalization_Perturbation_Plus(OCout)
            OCout_dec = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), size =(int(h/2),int(w/2)))))
        # OCoutenc_inter = F.relu(F.interpolate(self.OCenc_inter(OCout), size =(int(h/2),int(w/2))))
        # print(OCout_dec.shape)
        # print(hey)
        # print("---------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)

            OCout = F.relu(self.OC1_decbn(F.interpolate(self.OCdeclayer1(OCout_dec), size =(int(h/2),int(w/2)))))
            OCout = F.relu(self.OC2_decbn(F.interpolate(self.OCdeclayer2(OCout), scale_factor =(0.838,0.838))))
            OCout = F.relu(self.OC3_decbn(F.interpolate(self.OCdeclayer3(OCout), scale_factor =(0.798,0.798))))
            OCout = F.relu(self.OC4_decbn(F.interpolate(self.OCdeclayer4(OCout), size =(math.ceil(h/4),math.ceil(w/4)))))
        # # print(OCout.shape)
        # OCout_dec_inter = F.relu(F.interpolate(self.OCdec_inter(OCout), size =(math.ceil(h/4),math.ceil(w/4))))
        # print(OCout_dec_inter.shape)

        ##GMU fusion##
        # hv = self.tanh(self.hv(x))
        # ht = self.tanh(self.ht(OCout))
        # z = self.sigmoid(self.z(torch.cat([x,OCout],dim=1)))
        # x = z*hv + (1-z)*ht
        # print(x.shape)
        # print(OCout.shape)
        # print(hey)
        # if(training==True and p<0.5):
        #     x = torch.add(OCoutenc_inter, x)
        ##################
        x_tuple = self.layer1([x, w_arr])  # 400
        #f_map = x_tuple[0]
        # if(training==True and p2<0.5):
            # x_tuple[0] = self.Normalization_Perturbation_Plus(x_tuple[0])
        low_level = x_tuple[0]
        #print(low_level.shape)
        x_tuple = self.layer2(x_tuple)  # 100
        #print(x_tuple[0].shape)
        x_tuple = self.layer3(x_tuple)  # 100
        ##print(x_tuple[0].shape)
        x_tuple = self.layer4(x_tuple)  # 100
        #print(x_tuple[0].shape)
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        x_tsne = x_tuple[0]
        x = x_tuple[0]
        w_arr = x_tuple[1]

        x = self.aspp(x)
        #print(x.shape)
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        dec0_up = self.bot_aspp(x)
        # print(dec0_up.shape)

        dec0_fine = self.bot_fine(low_level)
        # print(dec0_fine.shape)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        # print(dec0_up.shape)
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        # print(dec0.shape)
        dec1 = self.final1(dec0)
        # print(dec1.shape)
        #dec1 = Upsample(dec1, (int(h/2),int(w/2)))
        # if(training==True and p3<0.5):
        #     dec1 = Upsample(dec1, (int(h/2),int(w/2)))
        #     # print("YOOOOOOOOOOOOOOOOOO")
        #     # print(dec1.shape)
        #     dec1 = torch.add(OCout_dec, dec1)

        
        #dec2 = torch.add(OCout, dec2)
        dec2 = self.final2(dec1)
        # print(dec2.shape)
        # print(hey)
        main_out = Upsample(dec2, x_size[2:])
        #print(main_out.shape)
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        #print(hey)
        if training:
            loss1 = self.criterion(main_out, gts)
            return_loss = loss1
            return return_loss
        else:
            return main_out, x_tsne


class SCFP(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,0,0,0,0,0], use_wtloss=False):
        super(SCFP, self).__init__()
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

        self.OClayer1 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1, bias=False).requires_grad_(False)
        self.OC1_bn = nn.BatchNorm2d(64).requires_grad_(False)
        #self.OC1_IN = nn.InstanceNorm2d(64,affine=False).requires_grad_(False)
        self.OClayer2 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1, bias=False).requires_grad_(False)
        self.OC2_bn = nn.BatchNorm2d(64).requires_grad_(False)
        #self.OC2_IN = nn.InstanceNorm2d(64,affine=False).requires_grad_(False)
        self.OClayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2, bias=False).requires_grad_(False)
        self.OC3_bn = nn.BatchNorm2d(64).requires_grad_(False)
        #self.OC3_IN = nn.InstanceNorm2d(128,affine=False).requires_grad_(False)
        self.OClayer4 = nn.Conv2d(64,256,kernel_size=3, stride=1, padding=2, dilation=2, bias=False).requires_grad_(False)
        self.OC4_bn = nn.BatchNorm2d(256).requires_grad_(False)

        self.OCdeclayer1 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1, bias=False).requires_grad_(False)
        self.OC1_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer2 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1, bias=False).requires_grad_(False)
        self.OC2_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2, bias=False).requires_grad_(False)
        self.OC3_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer4 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2, bias=False).requires_grad_(False)
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

        # self.OClayer2.weight = self.OClayer1.weight
        # self.OC2_bn.weight = self.OC1_bn.weight    
        # self.OClayer3.weight = self.OClayer2.weight
        # self.OC3_bn.weight = self.OC2_bn.weight

        # self.OCdeclayer1.weight = self.OClayer2.weight
        # self.OC1_decbn.weight = self.OC2_bn.weight
        # self.OCdeclayer2.weight = self.OClayer2.weight
        # self.OC2_decbn.weight = self.OC2_bn.weight
        # self.OCdeclayer3.weight = self.OClayer2.weight
        # self.OC3_decbn.weight = self.OC2_bn.weight
        # self.OCdeclayer4.weight = self.OCdeclayer2.weight
        # self.OC4_decbn.weight = self.OC2_decbn.weight

    
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
        # p3 = random.random()
        w_arr = []

        x_size = x.size()  # 800
        h,w = x_size[2:]

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

        if(training==True and p<0.5):
            
            OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp), size =(int(h/4),int(w/4)))))

            OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), size =(int(h/4),int(w/4)))))

            OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), size =(int(h/4),int(w/4)))))

            OCout_dec = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), size =(int(h/4),int(w/4)))))

            OCout = F.relu(self.OC1_decbn(F.interpolate(self.OCdeclayer1(OCout_dec), size =(int(h/4),int(w/4)))))
            OCout = F.relu(self.OC2_decbn(F.interpolate(self.OCdeclayer2(OCout), size =(int(h/4),int(w/4)))))
            OCout = F.relu(self.OC3_decbn(F.interpolate(self.OCdeclayer3(OCout), size =(int(h/4),int(w/4)))))
            OCout = F.relu(self.OC4_decbn(F.interpolate(self.OCdeclayer4(OCout), size =(math.ceil(h/4),math.ceil(w/4)))))

        if(training==True and p<0.5):
            x = torch.add(OCout, x)
        ##################
        x_tuple = self.layer1([x, w_arr])  # 400
        #f_map = x_tuple[0]
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
        #print(dec0_up.shape)
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        #print(dec0.shape)
        dec1 = self.final1(dec0)

        # if(training==True and p3<0.5):
        #     dec1 = Upsample(dec1, (int(h/2),int(w/2)))
        #     dec1 = torch.add(OCout_dec, dec1)

        dec2 = self.final2(dec1)

        main_out = Upsample(dec2, x_size[2:])

        if training:
            loss1 = self.criterion(main_out, gts)
            return_loss = loss1
            return return_loss
        else:
            return main_out


class dlrfp(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D16', wt_layer=[0,0,4,4,4,0,0], use_wtloss=False):
        super(dlrfp, self).__init__()
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
        self.OClayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC3_bn = nn.BatchNorm2d(64).requires_grad_(False)
        #self.OC3_IN = nn.InstanceNorm2d(128,affine=False).requires_grad_(False)
        self.OClayer4 = nn.Conv2d(64,256,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC4_bn = nn.BatchNorm2d(256).requires_grad_(False)

        self.OCdeclayer1 = nn.Conv2d(256,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC1_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer2 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        self.OC2_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC3_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        self.OCdeclayer4 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        self.OC4_decbn = nn.BatchNorm2d(64).requires_grad_(False)

        # self.OCdec_inter = nn.Conv2d(64,256,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        # self.OCenc_inter = nn.Conv2d(128,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)


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

        # initialize_weights_kaimingnormal_forOC(self.OCdec_inter)
        # initialize_weights_kaimingnormal_forOC(self.OCenc_inter)
    
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
        # if(training==True and p3<0.5):
        #     i = random.randint(2,b-1)
        #     ########################### APR ##################################
        #     NP_img = self.Normalization_Perturbation(torch.unsqueeze(x[i],0))
        #     # fft_1 = torch.fft.fftshift(torch.fft.rfftn(NP_img))

        #     # abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)
        #     # abs2 = abs_1 * 0.5

        #     # fft_2 = abs2*torch.exp((1j) * angle_1)

        #     # x_final2 = torch.fft.irfftn(torch.fft.ifftshift(fft_2))
        #     # x_final2[x_final2>255]=255
        #     # x_final2[x_final2<0]=0
            
        #     x = torch.cat([x[0:i], x[i+1:]])
        #     x = torch.cat([x,NP_img],dim=0)


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
        # initialize_weights_kaimingnormal_forOC(self.OCdec_inter)
        # initialize_weights_kaimingnormal_forOC(self.OCenc_inter)

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
        # if(training==True and p2<0.5):
            # x = self.Normalization_Perturbation_Plus(xp)
            # print(x)
            # print(hey)
            
        OCout = F.relu(self.OC1_bn(self.OClayer1(xp)))
        # print(OCout.shape)
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC2_bn(self.OClayer2(OCout)))
        # print(OCout.shape)
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC3_bn(self.OClayer3(OCout)))
        # print(OCout.shape)
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout_dec = F.relu(self.OC4_bn(self.OClayer4(OCout)))
        # OCoutenc_inter = F.relu(F.interpolate(self.OCenc_inter(OCout), size =(int(h/2),int(w/2))))
        # print(OCout_dec.shape)
        # print(hey)
        # print("---------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)

        OCout = F.relu(self.OC1_decbn(self.OCdeclayer1(OCout_dec)))
        OCout = F.relu(self.OC2_decbn(self.OCdeclayer2(OCout)))
        OCout = F.relu(self.OC3_decbn(self.OCdeclayer3(OCout)))
        OCout = F.relu(self.OC4_decbn(F.interpolate(self.OCdeclayer4(OCout), size =(math.ceil(h/4),math.ceil(w/4)))))
        # print(OCout.shape)
        # OCout_dec_inter = F.relu(F.interpolate(self.OCdec_inter(OCout), size =(math.ceil(h/4),math.ceil(w/4))))
        # print(OCout_dec_inter.shape)

        ##GMU fusion##
        # hv = self.tanh(self.hv(x))
        # ht = self.tanh(self.ht(OCout))
        # z = self.sigmoid(self.z(torch.cat([x,OCout],dim=1)))
        # x = z*hv + (1-z)*ht
        # print(x.shape)
        # print(OCout.shape)
        # print(hey)
        if(training==True and p<0.5):
            x = torch.add(OCout, x)
        ##################
        x_tuple = self.layer1([x, w_arr])  # 400
        #f_map = x_tuple[0]
        # if(training==True and p2<0.5):
            # x_tuple[0] = self.Normalization_Perturbation_Plus(x_tuple[0])
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
        # print(dec0_up.shape)

        dec0_fine = self.bot_fine(low_level)
        # print(dec0_fine.shape)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        # print(dec0_up.shape)
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        # print(dec0.shape)
        dec1 = self.final1(dec0)
        # if(training==True and p3<0.5):
            # dec1 = torch.add(OCout_dec_inter, dec1)
        # print(dec1.shape)
        #dec1 = Upsample(dec1, (int(h/2),int(w/2)))
        # if(training==True and p3<0.5):
            # dec1 = Upsample(dec1, (int(h/2),int(w/2)))
            # print("YOOOOOOOOOOOOOOOOOO")
            # print(dec1.shape)
            # dec1 = torch.add(OCout_dec, dec1)

        
        #dec2 = torch.add(OCout, dec2)
        dec2 = self.final2(dec1)
        # print(dec2.shape)
        # print(hey)
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






class ProRandConv(nn.Module):
    def __init__(self):
        super(ProRandConv, self).__init__()
        self.randconvblk = nn.Sequential(nn.Conv2d(3, 3, kernel_size=3, stride=1,padding=1,bias=False).requires_grad_(False),Norm2d(3).requires_grad_(False),nn.Tanh())
    def forward(self, x, L):
        for i in range(0,L):
            initialize_weights_kaimingnormal_forprorandconv(self.randconvblk)
            x=self.randconvblk(x)
        return x


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
        p = random.random()
        p2 = random.random()
        p3 = random.random()
        w_arr = []
        #x_hfi_bands = []
        x_size = x.size()  # 800
        h,w = x_size[2:]
        b,_,_,_ = x.shape
        # if(training==True and p3<0.5):
        #     i = random.randint(2,b-1)
        #     ########################### APR ##################################
        #     NP_img = self.Normalization_Perturbation(torch.unsqueeze(x[i],0))
        #     # fft_1 = torch.fft.fftshift(torch.fft.rfftn(NP_img))

        #     # abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)
        #     # abs2 = abs_1 * 0.5

        #     # fft_2 = abs2*torch.exp((1j) * angle_1)

        #     # x_final2 = torch.fft.irfftn(torch.fft.ifftshift(fft_2))
        #     # x_final2[x_final2>255]=255
        #     # x_final2[x_final2<0]=0
            
        #     x = torch.cat([x[0:i], x[i+1:]])
        #     x = torch.cat([x,NP_img],dim=0)


        if(training==True and p<0.5):
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
        if(training==True and p2<0.5):
            x = self.Normalization_Perturbation_Plus(xp)
            
        OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp),scale_factor =(1.205,1.205))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.2,1.2))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout_dec = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), size =(int(h/2),int(w/2)))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)

        OCout = F.relu(self.OC1_decbn(F.interpolate(self.OCdeclayer1(OCout_dec), size =(int(h/2),int(w/2)))))
        OCout = F.relu(self.OC2_decbn(F.interpolate(self.OCdeclayer2(OCout), scale_factor =(0.838,0.838))))
        OCout = F.relu(self.OC3_decbn(F.interpolate(self.OCdeclayer3(OCout), scale_factor =(0.798,0.798))))
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
        #f_map = x_tuple[0]
        if(training==True and p2<0.5):
            x_tuple[0] = self.Normalization_Perturbation_Plus(x_tuple[0])
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
        #x_hfl = []
        x_size = x.size()  # 800

        # ResNet
        x = self.layer0[0](x)
        #print(x.shape)
        x = self.layer0[1](x)
        #print(x.shape)
        x = self.layer0[2](x)
        #print(x.shape)
        x = self.layer0[3](x)


        x_tuple = self.layer1([x, w_arr])  # 400


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
        dec2 = self.final2(dec1)

        #print(dec2.shape)
        main_out = Upsample(dec2, x_size[2:])

        if training:
            loss1 = self.criterion(main_out, gts)
            return_loss = loss1
            return return_loss
        else:
            return main_out

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
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))
        
        # self.dsn = nn.Sequential(
        #     nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
        #     Norm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(0.1),
        #     nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # )

        
        #self.NPcls = Normalization_Perturbation_cls()

        #initialize_weights(self.NPcls)
        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        self.three_input_layer = False

        # self.prorandconv = ProRandConv()

        # self.xfm = DWTForward(J=6, mode='zero', wave='db3')
        # self.ixfm = DWTInverse(mode='zero', wave='db3')

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
        # print(x.shape)
        #print('gts shape',gts.shape)
        # print(torch.unique(x))
        # print(hey)
        p = random.random()
        w_arr = []
        # x_hfjh = []
        # x_hfj_bands = []
        # x_hfih = []
        # x_hfi_bands = []
        x_size = x.size()  # 800
        #b,_,_,_ = x.shape
        # print(x_size)
        # print(hey)
        # if(training==True):
        #     ########################## ProRandConv##################################
        #     pro_x = torch.cat([torch.unsqueeze(x[1],0),torch.unsqueeze(x[2],0),torch.unsqueeze(x[3],0),torch.unsqueeze(x[4],0)],dim=0)
        #     pro_x = self.prorandconv(pro_x,L)
        #     #print(pro_x.shape)
        #     x = torch.cat([x[0:1], x[2:]])
        #     x = torch.cat([x[0:2], x[3:]])
        #     x = torch.cat([x[0:3], x[4:]])
        #     x = torch.cat([x[0:4], x[5:]])
        #     x = torch.cat([x,pro_x],dim=0)
        #     ##################################################################
        # ResNet
        x = self.layer0[0](x)
        #print(x.shape)
        x = self.layer0[1](x)
        #print(x.shape)
        x = self.layer0[2](x)
        #print(x.shape)
        x = self.layer0[3](x)
        # print(x.shape)
        # x_np = torch.squeeze(x).cpu().detach().numpy()
        # fft_image = np.fft.fft2(x_np)
        # fft_image = np.fft.fftshift(fft_image)  # Shift the zero frequency component to the center
        # magnitude_spectrum = np.abs(fft_image)
        # magnitude_spectrum = magnitude_spectrum/np.max(magnitude_spectrum)
        # a = 0.3  # Adjust as needed
        # b = 0.7 # Adjust as needed
        # low_band_mask = (magnitude_spectrum >= 0) & (magnitude_spectrum < a)
        # mid_band_mask = (magnitude_spectrum >= a) & (magnitude_spectrum < b)
        # high_band_mask = magnitude_spectrum >= b
        # low_band_count = np.sum(low_band_mask)
        # mid_band_count = np.sum(mid_band_mask)
        # high_band_count = np.sum(high_band_mask)
        # print("Stage 1 --------------->>>>>>>>>>>>>>>>")
        # print(low_band_count)
        # print(mid_band_count)
        # print(high_band_count)
        # bands = ['Low Band', 'Mid Band', 'High Band']
        # counts = [low_band_count, mid_band_count, high_band_count]
        # print(counts)
        # plt.bar(bands, counts)
        # plt.title('Frequency Band Counts')
        # plt.ylabel('Number of Values')
        # plt.savefig("./baseline_deeplabv3_stage1_hist_thresh.jpg")
        # print(hey)
        # if(training==True and p<0.5):
        # xp = x
        # x = self.Normalization_Perturbation_Plus(x)
        # xNP = x
        x_tuple = self.layer1([x, w_arr])  # 400
        # print(x_tuple[0].shape)
        # # print(hey)
        # x_np = torch.squeeze(x_tuple[0]).cpu().detach().numpy()
        # fft_image = np.fft.fft2(x_np)
        # fft_image = np.fft.fftshift(fft_image)  # Shift the zero frequency component to the center
        # magnitude_spectrum = np.abs(fft_image)
        # magnitude_spectrum = magnitude_spectrum/np.max(magnitude_spectrum)
        # a = 0.3  # Adjust as needed
        # b = 0.7 # Adjust as needed
        # low_band_mask = (magnitude_spectrum >= 0) & (magnitude_spectrum < a)
        # mid_band_mask = (magnitude_spectrum >= a) & (magnitude_spectrum < b)
        # high_band_mask = magnitude_spectrum >= b
        # low_band_count = np.sum(low_band_mask)
        # mid_band_count = np.sum(mid_band_mask)
        # high_band_count = np.sum(high_band_mask)
        # print("Stage 2 --------------------->>>>>>>>>>>>")
        # print(low_band_count)
        # print(mid_band_count)
        # print(high_band_count)
        # bands = ['Low Band', 'Mid Band', 'High Band']
        # counts = [low_band_count, mid_band_count, high_band_count]
        # print(counts)
        # plt.bar(bands, counts)
        # plt.title('Frequency Band Counts')
        # plt.ylabel('Number of Values')
        # plt.savefig("./baseline_deeplabv3_stage2_hist_thresh.jpg")
        # print(hey)

        # if(training==True and p<0.5):
        #    x_tuple[0] = self.Normalization_Perturbation_Plus(x_tuple[0])

        low_level = x_tuple[0]
        #print(low_level.shape)
        x_tuple = self.layer2(x_tuple)  # 100
        # print(x_tuple[0].shape)
        x_tuple = self.layer3(x_tuple)  # 100
        # print(x_tuple[0].shape)
        x_tuple = self.layer4(x_tuple)  # 100
        x_tsne = x_tuple[0]
        # print(x_tuple[0].shape)
        
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        
        x = x_tuple[0]
        w_arr = x_tuple[1]

        x = self.aspp(x)
        # print(x.shape)
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        dec0_up = self.bot_aspp(x)
        # print(dec0_up.shape)

        dec0_fine = self.bot_fine(low_level)
        #print(dec0_fine.shape)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        #print(dec0_up.shape)
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        #print(dec0.shape)
        dec1 = self.final1(dec0)
        #print(dec1.shape)
        dec2 = self.final2(dec1)

        #print(dec2.shape)
        main_out = Upsample(dec2, x_size[2:])
        # print(hey)
        #print(main_out.shape)
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        #print(hey)
        if training:
            loss1 = self.criterion(main_out, gts)
            return_loss = loss1
            return return_loss
        else:
            return main_out, x_tsne        

class oldDeepMAO(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "efficientnet-b3",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = False,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 3
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
                                
        classes = 3
        
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

        self.OClayer1 = nn.Conv2d(64,56,kernel_size=3, stride=1, padding=1)
        self.OC1_bn = nn.BatchNorm2d(56)
        self.OClayer2 = nn.Conv2d(56,64,kernel_size=3, stride=1, padding=1)
        self.OC2_bn = nn.BatchNorm2d(64)
        self.OClayer3 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC3_bn = nn.BatchNorm2d(128)
        self.OClayer4 = nn.Conv2d(128,16,kernel_size=3, stride=1, padding=2, dilation=2)
        self.OC4_bn = nn.BatchNorm2d(16)

        #self.SEblock = torchvision.ops.SqueezeExcitation(16,16)
        

 
    def forward(self, x):
        
        features = self.encoder(x)
        # print('features',len(features))
        # print('features',features[1].shape)

        # for i in range(6):
        #     print('features',features[i].shape)
        
        # OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(features[1]),scale_factor =(1.1,1.1)))) #layersize256 #output320
        # print('OCout---',OCout.shape)
        
        # _,_,h1,w1 = features[0].shape #512
        
        # OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.1,1.1))))#layersize320 #output400
        # print('OCout22---',OCout.shape)

        # OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.1,1.1))))#layersize400 output500
        # print('OCout-33--',OCout.shape)

        # OCout = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), scale_factor =(1.1,1.1))))#layersize500 output625
        # print('OCout--44-',OCout.shape)

        # OCout = F.interpolate(OCout, size = (h1,w1))#625 to 512
        # print('OCout---',OCout.shape)

        logit = self.decoder(*features) #16
        print(hey)
        
        _,_,h,w = logit.shape
        
        
        if(logit.shape==OCout.shape):
            logit = torch.add(OCout, logit)
        else:
            OCout = F.interpolate(OCout,size=(h,w),mode='bilinear')
            logit = torch.add(OCout, logit)

        logit = self.segmentation_head(logit)
        
        return logit
    

class DeepMAO(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "efficientnet-b3",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = False,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 3
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
                                
        classes = 3
        
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

        #self.SEblock = torchvision.ops.SqueezeExcitation(16,16)
        
    def forward(self, x):
        
        features = self.encoder(x)
        
        OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(features[1]),scale_factor =(1.2,1.2)))) #layersize256 #output320
        
        _,_,h1,w1 = features[0].shape #512
        #print(h1,w1)
        OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))#layersize320 #output400
        #print(OCout.shape)

        
        OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.2,1.2))))#layersize400 output500
        #print(OCout.shape)
        OCout = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), scale_factor =(1.15,1.15))))#layersize500 output625
        #print(OCout.shape)

        
        OCout = F.interpolate(OCout, size = (h1,w1))#625 to 512
        
        #OCout = self.SEblock(OCout)

        logit = self.decoder(*features) #16
        
        _,_,h,w = logit.shape
        
        
        if(logit.shape==OCout.shape):
            logit = torch.add(OCout, logit)
        else:
            OCout = F.interpolate(OCout,size=(h,w),mode='bilinear')
            logit = torch.add(OCout, logit)

        logit = self.segmentation_head(logit)
        
        return logit

class DeepLabV3plus(smp.DeepLabV3Plus):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, x):
        features = self.encoder(x)
        # print(list(map(lambda x: x.shape, features)))
        decoder_output = self.decoder(*features)
        # print(decoder_output.shape) # .x256x128x128
        logit = self.segmentation_head(decoder_output)
        # multi_features = features[-2:] + [decoder_output]
        # multi_features = features[-3:] + [decoder_output]
        # multi_features = features[-4:] + [decoder_output]
        # multi_features = features[-5:] + [decoder_output]
        # multi_features = features[-1:] + [decoder_output]
        multi_features = [decoder_output]
        return logit
    
# class DeepMAO(SegmentationModel):
#     def __init__(
#         self,
#         encoder_name: str = "efficientnet-b3",
#         encoder_depth: int = 5,
#         encoder_weights: Optional[str] = "imagenet",
#         decoder_use_batchnorm: bool = False,
#         decoder_channels: List[int] = (256, 128, 64, 32, 16),
#         in_channels: int = 3,
#         classes: int = 3
#     ):
#         super().__init__()

#         self.encoder = get_encoder(
#             encoder_name,
#             in_channels=in_channels,
#             depth=encoder_depth,
#             weights=encoder_weights,
#         )
                                
#         classes = 3


class Unet_dummy(SegmentationModel):
    # def __init__(self,name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):
    #     super().__init__()
    def __init__(
        self,
        encoder_name: str = "efficientnet-b3",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = False,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 3
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
                                
        # classes = 3

        #self.enc = EfficientNet(arch=arch, num_classes=9)
        # self.dec = UnetDecoder(
        #     encoder_channels=[40,32,48,136,384],
        #     out_channels=256,
        #     atrous_rates=[12,24,36],
        #     output_stride=16)
        
        self.dec = UnetDecoder(
            encoder_channels=(3,40,32,48,136,384),
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
        )

        self.seg_head = SegmentationHead(16, 9, activation=None, upsampling=4)

    def forward(self, x):
        features = self.encoder(x)
        #rint("rgb",features[-1][0].shape, features[-1][1].shape, features[-4][1].shape)
        #for i in range(len(features)):
        #    print('shapes of features in EO Train 1',features[i].shape)
        #    print(features[i].shape)
        decoder_output = self.dec(*features)
        # print(decoder_output.shape) # .x256x128x128
        logit = self.seg_head(decoder_output)
        multi_features = features[-2:] + [decoder_output]
        # multi_features = features[-3:] + [decoder_output]
        # multi_features = features[-4:] + [decoder_output]
        # multi_features = features[-5:] + [decoder_output]
        # multi_features = features[-1:] + [decoder_output]
        #multi_features = features[-2:] + [decoder_output]
        #
        #multi_features = features[-2:]
        #print('multi-features 0 ',(multi_features[0].shape))
        #print('multi-features 1 ',(multi_features[1].shape))
        #
        return logit
    

class Unet_ibnnet(SegmentationModel):
    # def __init__(self,name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs): 0 0 4 4 4 0 0
    #     super().__init__()
    def __init__(
        self, criterion,
        # encoder_name: str = "resnet50",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = False,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 3, trunk='resnet-50',variant='D16',wt_layer=[0,0,4,4,4,0,0],
    ):  ###wt_layer set for ibn-net. 
        super().__init__()

        # self.encoder = get_encoder(
        #     encoder_name,
        #     in_channels=in_channels,
        #     depth=encoder_depth,
        #     weights=encoder_weights,
        # )
        self.trunk = trunk
        self.variant = variant
        self.wt_layer = wt_layer
        self.criterion = criterion

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
                                
        # classes = 3

        #self.enc = EfficientNet(arch=arch, num_classes=9)
        # self.dec = UnetDecoder(
        #     encoder_channels=[40,32,48,136,384],
        #     out_channels=256,
        #     atrous_rates=[12,24,36],
        #     output_stride=16)
        ##(3,40,32,48,136,384) -> encoder_channels
        self.dec = UnetDecoder(
            encoder_channels=(3,64,256,512,1024,2048),
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
        )

        # self.seg_head = SegmentationHead(16, classes , activation=None)#, upsampling=4)
        self.seg_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=  classes, upsampling=2 )# , activation=None)
        ########################################### Stage 1 ##################################################################
        # self.OClayer1 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        # self.OC1_bn = nn.BatchNorm2d(64).requires_grad_(False)
        # self.OClayer2 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        # self.OC2_bn = nn.BatchNorm2d(64).requires_grad_(False)
        # self.OClayer3 = nn.Conv2d(64,128,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        # self.OC3_bn = nn.BatchNorm2d(128).requires_grad_(False)
        # self.OClayer4 = nn.Conv2d(128,256,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        # self.OC4_bn = nn.BatchNorm2d(256).requires_grad_(False)

        # self.OCdeclayer1 = nn.Conv2d(256,128,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        # self.OC1_decbn = nn.BatchNorm2d(128).requires_grad_(False)
        # self.OCdeclayer2 = nn.Conv2d(128,64,kernel_size=3, stride=1, padding=1).requires_grad_(False)
        # self.OC2_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        # self.OCdeclayer3 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        # self.OC3_decbn = nn.BatchNorm2d(64).requires_grad_(False)
        # self.OCdeclayer4 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=2, dilation=2).requires_grad_(False)
        # self.OC4_decbn = nn.BatchNorm2d(64).requires_grad_(False)

        # initialize_weights_kaimingnormal_forOC(self.OClayer1)
        # initialize_weights_kaimingnormal_forOC(self.OC1_bn)
        # initialize_weights_kaimingnormal_forOC(self.OClayer2)
        # initialize_weights_kaimingnormal_forOC(self.OC2_bn)
        # initialize_weights_kaimingnormal_forOC(self.OClayer3)
        # initialize_weights_kaimingnormal_forOC(self.OC3_bn)
        # initialize_weights_kaimingnormal_forOC(self.OClayer4)
        # initialize_weights_kaimingnormal_forOC(self.OC4_bn)
        # initialize_weights_kaimingnormal_forOC(self.OCdeclayer1)
        # initialize_weights_kaimingnormal_forOC(self.OC1_decbn)
        # initialize_weights_kaimingnormal_forOC(self.OCdeclayer2)
        # initialize_weights_kaimingnormal_forOC(self.OC2_decbn)
        # initialize_weights_kaimingnormal_forOC(self.OCdeclayer3)
        # initialize_weights_kaimingnormal_forOC(self.OC3_decbn)
        # initialize_weights_kaimingnormal_forOC(self.OCdeclayer4)
        # initialize_weights_kaimingnormal_forOC(self.OC4_decbn)

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
    
        # initialize_weights(self.aspp)
        # initialize_weights(self.bot_aspp)
        # initialize_weights(self.bot_fine)
        # initialize_weights(self.final1)
        # initialize_weights(self.final2)

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

    def forward(self, x, gts=None, training=True):
        features = []
        w_arr = []
        # print('hola')
        #x_hfl = []
        x_size = x.size()  # 800
        # features.append[x]
        features.append(x)

        # ResNet
        x = self.layer0[0](x)
        # print('x 0 ---',x.shape)
        # features.append(x)
        #print(x.shape)
        x = self.layer0[1](x)
        # print('x 1 ---',x.shape)
        # features.append(x)
        #print(x.shape)
        x = self.layer0[2](x)
        # print('x 2 ---',x.shape)
        # features.append(x)
        #print(x.shape)
        x = self.layer0[3](x)
        # print('x 3 ---',x.shape)
        features.append(x)

        x_tuple = self.layer1([x, w_arr])  # 400
        features.append(x_tuple[0])

        low_level = x_tuple[0]
        #print(low_level.shape)
        x_tuple = self.layer2(x_tuple)  # 100
        features.append(x_tuple[0])
        #print(x_tuple[0].shape)
        x_tuple = self.layer3(x_tuple)  # 100
        features.append(x_tuple[0])
        # print('x_tuple[0] -------------',x_tuple[0].shape)
        x_tuple = self.layer4(x_tuple)  # 100
        features.append(x_tuple[0])
        # print('x_tuple---',len(x_tuple))
        # print('x_tuple[0] ---------------after',x_tuple[0].shape)
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        
        # x = x_tuple[0]
        # w_arr = x_tuple[1]
        # print('shape-------------',x.shape)
        # print('shape-------------',len(features))

        for i in range(len(features)):
            print('shapes of features in EO Train ',features[i].shape)

        # print(hey)
        # features = self.encoder(x)
        # print('len of features',len(features))
        # #rint("rgb",features[-1][0].shape, features[-1][1].shape, features[-4][1].shape)
        # for i in range(len(features)):
        #    print('shapes of features in EO Train 1',features[i].shape)
        #    print(features[i].shape)

        # OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(features[1]),scale_factor =(1.2,1.2)))) #layersize256 #output320
        
        # _,_,h1,w1 = features[0].shape #512
        # #print(h1,w1)
        # OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))#layersize320 #output400
        # #print(OCout.shape)

        
        # OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.2,1.2))))#layersize400 output500
        # #print(OCout.shape)
        # OCout = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), scale_factor =(1.15,1.15))))#layersize500 output625
        # #print(OCout.shape)

        
        # OCout = F.interpolate(OCout, size = (h1,w1))#625 to 512

        logit = self.dec(*features)
        # print('logit------------------',logit.shape)
        # _,_,h,w = logit.shape
        
        
        # if(logit.shape==OCout.shape):
        #     logit = torch.add(OCout, logit)
        # else:
        #     OCout = F.interpolate(OCout,size=(h,w),mode='bilinear')
        #     logit = torch.add(OCout, logit)


        # print(decoder_output.shape) # .x256x128x128
        # print('logit---------',logit.shape)
        main_out = self.seg_head(logit)


        # print('logit---------after',logit.shape)
        # multi_features = features[-2:] + [decoder_output]
        # multi_features = features[-3:] + [decoder_output]
        # multi_features = features[-4:] + [decoder_output]
        # multi_features = features[-5:] + [decoder_output]
        # multi_features = features[-1:] + [decoder_output]
        #multi_features = features[-2:] + [decoder_output]
        #
        #multi_features = features[-2:]
        #print('multi-features 0 ',(multi_features[0].shape))
        #print('multi-features 1 ',(multi_features[1].shape))

                #dec2 = torch.add(OCout, dec2)
        # dec2 = self.final2(dec1)
        # #print(dec2.shape)
        # main_out = Upsample(dec2, x_size[2:])
        #print(main_out.shape)
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        #print(hey)
        if training:
            loss1 = self.criterion(main_out, gts)
            return_loss = loss1
            # print(hey)
            # print('unet')
            return return_loss
        else:

            return main_out #, x_tsne
        # print(hey)
        # return logit



class Unet_MRFP(SegmentationModel):
    # def __init__(self,name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs): 0 0 4 4 4 0 0
    #     super().__init__()
    def __init__(
        self,criterion,
        # encoder_name: str = "resnet50",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = False,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 3, trunk='resnet-50',variant='D16',wt_layer=[0,0,4,4,4,0,0]
    ):  ###wt_layer set for ibn-net. 
        super().__init__()

        # self.encoder = get_encoder(
        #     encoder_name,
        #     in_channels=in_channels,
        #     depth=encoder_depth,
        #     weights=encoder_weights,
        # )
        self.trunk = trunk
        self.variant = variant
        self.wt_layer = wt_layer
        self.criterion = criterion

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
                                
        # classes = 3

        #self.enc = EfficientNet(arch=arch, num_classes=9)
        # self.dec = UnetDecoder(
        #     encoder_channels=[40,32,48,136,384],
        #     out_channels=256,
        #     atrous_rates=[12,24,36],
        #     output_stride=16)
        ##(3,40,32,48,136,384) -> encoder_channels
        self.dec = UnetDecoder(
            encoder_channels=(3,64,256,512,1024,2048),
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
        )

        self.seg_head = SegmentationHead(16, classes , activation=None, upsampling=2)

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

        ################################################ Stage 2 #########################################
        #########################################################################################################
    
        # initialize_weights(self.aspp)
        # initialize_weights(self.bot_aspp)
        # initialize_weights(self.bot_fine)
        # initialize_weights(self.final1)
        # initialize_weights(self.final2)

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
    
    def forward(self, x, gts=None, training=True):
    # def forward(self, x, training=True):
        features = []
        p = random.random()
        p2 = random.random()
        p3 = random.random()
        w_arr = []
        x_size = x.size()  # 800
        h,w = x_size[2:]
        b,_,_,_ = x.shape
        # print('hola')
        #x_hfl = []
        x_size = x.size()  # 800
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


                # features.append[x]
        features.append(x)

        # ResNet
        x = self.layer0[0](x)
        # print('x 0 ---',x.shape)
        # features.append(x)
        #print(x.shape)
        x = self.layer0[1](x)
        # print('x 1 ---',x.shape)
        # features.append(x)
        #print(x.shape)
        x = self.layer0[2](x)
        # print('x 2 ---',x.shape)
        # features.append(x)
        #print(x.shape)
        x = self.layer0[3](x)
        # print('x 3 ---',x.shape)
        xp = x
        if(training==True and p2<0.5):
            x = self.Normalization_Perturbation_Plus(xp)

        OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(xp),scale_factor =(1.205,1.205))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.2,1.2))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)
        OCout_dec = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), size =(int(h/2),int(w/2)))))
        # if(training==True and p<0.5):
        #     OCout = self.Normalization_Perturbation_Plus(OCout)

        OCout = F.relu(self.OC1_decbn(F.interpolate(self.OCdeclayer1(OCout_dec), size =(int(h/2),int(w/2)))))
        OCout = F.relu(self.OC2_decbn(F.interpolate(self.OCdeclayer2(OCout), scale_factor =(0.838,0.838))))
        OCout = F.relu(self.OC3_decbn(F.interpolate(self.OCdeclayer3(OCout), scale_factor =(0.798,0.798))))
        OCout = F.relu(self.OC4_decbn(F.interpolate(self.OCdeclayer4(OCout), size =(math.ceil(h/4),math.ceil(w/4)))))
        
        if(training==True and p<0.5):
            # print('OCout--------------',OCout.shape)
            # print('x--------------',x.shape)
            x = torch.add(OCout, x)
        features.append(x)
        ##GMU fusion##
        # hv = self.tanh(self.hv(x))
        # ht = self.tanh(self.ht(OCout))
        # z = self.sigmoid(self.z(torch.cat([x,OCout],dim=1)))
        # x = z*hv + (1-z)*ht
        x_tuple = self.layer1([x, w_arr]) 
        if(training==True and p2<0.5):
            x_tuple[0] = self.Normalization_Perturbation_Plus(x_tuple[0])
        features.append(x_tuple[0])
        # print(hey)

        low_level = x_tuple[0]
        #print(low_level.shape)
        x_tuple = self.layer2(x_tuple)  # 100
        features.append(x_tuple[0])
        #print(x_tuple[0].shape)
        x_tuple = self.layer3(x_tuple)  # 100
        features.append(x_tuple[0])
        # print('x_tuple[0] -------------',x_tuple[0].shape)
        x_tuple = self.layer4(x_tuple)  # 100
        features.append(x_tuple[0])
        # print('x_tuple---',len(x_tuple))
        # print('x_tuple[0] ---------------after',x_tuple[0].shape)
        #print("-------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>")
        
        # x = x_tuple[0]
        # w_arr = x_tuple[1]
        # print('shape-------------',x.shape)
        # print('shape-------------',len(features))

        # for i in range(len(features)):
        #     print('shapes of features in EO Train ',features[i].shape)

        # print(hey)
        # features = self.encoder(x)
        # print('len of features',len(features))
        # #rint("rgb",features[-1][0].shape, features[-1][1].shape, features[-4][1].shape)
        # for i in range(len(features)):
        #    print('shapes of features in EO Train 1',features[i].shape)
        #    print(features[i].shape)

        # OCout = F.relu(self.OC1_bn(F.interpolate(self.OClayer1(features[1]),scale_factor =(1.2,1.2)))) #layersize256 #output320
        
        # _,_,h1,w1 = features[0].shape #512
        # #print(h1,w1)
        # OCout = F.relu(self.OC2_bn(F.interpolate(self.OClayer2(OCout), scale_factor =(1.2,1.2))))#layersize320 #output400
        # #print(OCout.shape)

        
        # OCout = F.relu(self.OC3_bn(F.interpolate(self.OClayer3(OCout), scale_factor =(1.2,1.2))))#layersize400 output500
        # #print(OCout.shape)
        # OCout = F.relu(self.OC4_bn(F.interpolate(self.OClayer4(OCout), scale_factor =(1.15,1.15))))#layersize500 output625
        # #print(OCout.shape)

        
        # OCout = F.interpolate(OCout, size = (h1,w1))#625 to 512

        logit = self.dec(*features)

        # _,_,h,w = logit.shape
        
        
        # if(logit.shape==OCout.shape):
        #     logit = torch.add(OCout, logit)
        # else:
        #     OCout = F.interpolate(OCout,size=(h,w),mode='bilinear')
        #     logit = torch.add(OCout, logit)


        # print(decoder_output.shape) # .x256x128x128
        # print('logit---------',logit.shape)
        main_out = self.seg_head(logit)
        # print('logit---------after',logit.shape)
        # print(hey)
        # multi_features = features[-2:] + [decoder_output]
        # multi_features = features[-3:] + [decoder_output]
        # multi_features = features[-4:] + [decoder_output]
        # multi_features = features[-5:] + [decoder_output]
        # multi_features = features[-1:] + [decoder_output]
        #multi_features = features[-2:] + [decoder_output]
        #
        #multi_features = features[-2:]
        #print('multi-features 0 ',(multi_features[0].shape))
        #print('multi-features 1 ',(multi_features[1].shape))
        if training:
            loss1 = self.criterion(main_out, gts)
            return_loss = loss1
            # print(hey)
            # print('unet')
            return return_loss
        else:

            return main_out #, x_tsne
        #
        # return logit
    
