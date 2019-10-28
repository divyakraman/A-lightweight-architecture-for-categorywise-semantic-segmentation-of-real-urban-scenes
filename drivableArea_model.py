import numpy as np 
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
#import matplotlib.pyplot as plt  
#import scipy.misc as smisc
#import random


class FPN(nn.Module):
	def __init__(self):
		 super(FPN, self).__init__()
		 self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.down1 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv1_bn = nn.BatchNorm2d(32)
		 self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.down2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv2_bn = nn.BatchNorm2d(64)
		 self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.down3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=4,stride=2,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv3_bn = nn.BatchNorm2d(128)
		 self.lateral3 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.lateral2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.lateral1 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.upsample3 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
		 self.upsample2 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
		 self.newft3_bn = nn.BatchNorm2d(64)
		 self.newft2_bn = nn.BatchNorm2d(64)
		 self.newft1_bn = nn.BatchNorm2d(64)



	def forward(self,input):
		ft1 = self.conv1_bn(F.leaky_relu(self.down1(F.leaky_relu(self.conv1(input))))) #1*32*160*160
		ft2 = self.conv2_bn(F.leaky_relu(self.down2(F.leaky_relu(self.conv2(ft1)))))  #1*64*80*80
		ft3 = self.conv3_bn(F.leaky_relu(self.down3(F.leaky_relu(self.conv3(ft2)))))  #1*128*40*40  
		new_ft3 = self.newft3_bn(self.lateral3(ft3)) #1*64*40*40 ; name as lateral3
		new_ft2 = self.newft2_bn(self.lateral2(ft2)+self.upsample3(new_ft3)) #1*64*80*80
		new_ft1 = self.newft1_bn(self.lateral1(ft1)+self.upsample2(new_ft2)) #1*64*160*160
		return new_ft1,new_ft2,new_ft3



class SA(nn.Module):
	def __init__(self):
		 super(SA, self).__init__()
		 self.downsample = nn.Conv2d(in_channels=192,out_channels=128,kernel_size=4,stride=2,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv1 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.upsample3 = nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True)
		 self.upsample2 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
		 self.downsample_bn = nn.BatchNorm2d(128)
		 self.bn_conv1 = nn.BatchNorm2d(128)
		 self.bn_conv2 = nn.BatchNorm2d(128)
		 self.bn_conv3 = nn.BatchNorm2d(128)
		 self.bn_conv4 = nn.BatchNorm2d(128)


		 

	def forward(self,new_ft1,new_ft2,new_ft3):
		sa1 = new_ft1
		sa2 = self.upsample2(new_ft2)
		sa3 = self.upsample3(new_ft3)
		concat = torch.cat((sa1,sa2,sa3),dim=1)	#1*96*80*80	#Difference between torch.stack and torch.cat
		downsample = self.downsample_bn(F.leaky_relu(self.downsample(concat))) #1*128*80*80
		conv = self.bn_conv1(F.leaky_relu(self.conv1(downsample))) #1*128*80*80
		conv = self.bn_conv2(F.leaky_relu(self.conv2(conv))) #1*128*80*80
		downsample = conv+downsample #residual connection; 1*128*80*80
		conv = self.bn_conv3(F.leaky_relu(self.conv3(downsample))) #1*128*80*80
		conv = self.bn_conv4(F.leaky_relu(self.conv4(conv))) #1*128*80*80
		out = conv+downsample #residual connection; 1*128*80*80
		return out

class globalpool(nn.Module):
	def __init__(self):
		super(globalpool, self).__init__()    
		self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv1_bn = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv2_bn = nn.BatchNorm2d(16)
		self.conv3 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv3_bn = nn.BatchNorm2d(16)
		self.downsample1 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.downsample1_bn = nn.BatchNorm2d(32) #1*32*160*160
		self.conv4 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv5 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv4_bn = nn.BatchNorm2d(32)
		self.conv5_bn = nn.BatchNorm2d(32)
		self.downsample2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.downsample2_bn = nn.BatchNorm2d(64) #1*64*80*80
		self.conv6 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv7 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv6_bn = nn.BatchNorm2d(64)
		self.conv7_bn = nn.BatchNorm2d(64)
		self.conv8 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv9 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv8_bn = nn.BatchNorm2d(64)
		self.conv9_bn = nn.BatchNorm2d(64)

	def forward(self,input,object_sa_maps):
		#input is rgb image to compute global maps.
		layer1 = self.conv1_bn(F.leaky_relu(self.conv1(input)))
		global_maps = self.conv2_bn(F.leaky_relu(self.conv2(layer1))) #1*16*320*320
		global_maps = self.conv3_bn(F.leaky_relu(self.conv3(global_maps))) #1*16*320*320
		global_maps = global_maps + layer1 #1*16*320*320
		global_maps = self.downsample1_bn(F.leaky_relu(self.downsample1(global_maps)))#1*32*160*160
		layer1 = self.conv4_bn(F.leaky_relu(self.conv4(global_maps)))
		layer1 = self.conv5_bn(F.leaky_relu(self.conv5(global_maps)))
		global_maps = global_maps+layer1
		global_maps = self.downsample2_bn(F.leaky_relu(self.downsample2(global_maps)))#1*64*80*80
		layer1 = self.conv6_bn(F.leaky_relu(self.conv6(global_maps)))
		layer1 = self.conv7_bn(F.leaky_relu(self.conv7(global_maps)))
		global_maps = global_maps+layer1
		layer1 = self.conv8_bn(F.leaky_relu(self.conv8(global_maps)))
		layer1 = self.conv9_bn(F.leaky_relu(self.conv9(global_maps)))
		global_maps = global_maps+layer1
		out = torch.cat((global_maps,object_sa_maps),dim=1)  #1*192*80*80
		return out


class dilatedResidualNetworks(nn.Module):
	def __init__(self,num_classes=7):
		super(dilatedResidualNetworks,self).__init__()
		self.downsample1 = nn.Conv2d(in_channels=192,out_channels=256,kernel_size=4,stride=2,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.downsample1_bn = nn.BatchNorm2d(256)
		#Dilation 1, 2 residual blocks
		self.conv1 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv1_bn = nn.BatchNorm2d(256)
		self.conv2_bn = nn.BatchNorm2d(256)
		self.conv3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv3_bn = nn.BatchNorm2d(256)
		self.conv4_bn = nn.BatchNorm2d(256)
		#Dilation 2, 2 residual blocks
		self.conv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=2,dilation=2,groups=1,bias=True,padding_mode='zeros')
		self.conv6 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=2,dilation=2,groups=1,bias=True,padding_mode='zeros')
		self.conv5_bn = nn.BatchNorm2d(256)
		self.conv6_bn = nn.BatchNorm2d(256)
		self.conv7 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=2,dilation=2,groups=1,bias=True,padding_mode='zeros')
		self.conv8 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=2,dilation=2,groups=1,bias=True,padding_mode='zeros')
		self.conv7_bn = nn.BatchNorm2d(256)
		self.conv8_bn = nn.BatchNorm2d(256)
		#Add more channels
		self.deeper = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.deeper_bn = nn.BatchNorm2d(512)
		#Dilation 4, 2 residual blocks
		self.conv9 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=4,dilation=4,groups=1,bias=True,padding_mode='zeros')
		self.conv10 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=4,dilation=4,groups=1,bias=True,padding_mode='zeros')
		self.conv9_bn = nn.BatchNorm2d(512)
		self.conv10_bn = nn.BatchNorm2d(512)
		self.conv11 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=4,dilation=4,groups=1,bias=True,padding_mode='zeros')
		self.conv12 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=4,dilation=4,groups=1,bias=True,padding_mode='zeros')
		self.conv11_bn = nn.BatchNorm2d(512)
		self.conv12_bn = nn.BatchNorm2d(512)
		#Dilation 2, 2 conv blocks
		self.conv13 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=2,dilation=2,groups=1,bias=True,padding_mode='zeros')
		self.conv14 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=2,dilation=2,groups=1,bias=True,padding_mode='zeros')
		self.conv13_bn = nn.BatchNorm2d(512)
		self.conv14_bn = nn.BatchNorm2d(512)
		#Dilation 1, 2 conv blocks
		self.conv15 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv16 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv15_bn = nn.BatchNorm2d(512)
		self.conv16_bn = nn.BatchNorm2d(512)
		#Generate segmentation maps
		self.seg = nn.Conv2d(in_channels=512,out_channels=num_classes,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.softmax = nn.LogSoftmax()
		#Initializing final layer
		m = self.seg
		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
		m.weight.data.normal_(0, math.sqrt(2. / n))
		m.bias.data.zero_()
		self.upsample = nn.Upsample(scale_factor=8,mode='bilinear',align_corners=True)




	def forward(self,input,num_classes=7):
		#input is global and multi scale feature maps
		out = self.downsample1_bn(F.leaky_relu(self.downsample1(input))) #1*256*40*40
		#Dilation 1, 2 residual blocks
		layer1 = self.conv1_bn(F.leaky_relu(self.conv1(out))) 
		layer1 = self.conv2_bn(F.leaky_relu(self.conv2(layer1)))
		out=out+layer1
		layer1 = self.conv3_bn(F.leaky_relu(self.conv3(out)))
		layer1 = self.conv4_bn(F.leaky_relu(self.conv4(layer1)))
		out=out+layer1
		#Dilation 2, 2 residual blocks
		layer1 = self.conv5_bn(F.leaky_relu(self.conv5(out)))
		layer1 = self.conv6_bn(F.leaky_relu(self.conv6(layer1)))
		out=out+layer1
		layer1 = self.conv7_bn(F.leaky_relu(self.conv7(out)))
		layer1 = self.conv8_bn(F.leaky_relu(self.conv8(layer1)))
		out=out+layer1
		#Let's go deeper
		out = self.deeper_bn(F.leaky_relu(self.deeper(out)))
		#Dilation 4, 2 residual blocks
		layer1 = self.conv9_bn(F.leaky_relu(self.conv9(out))) 
		layer1 = self.conv10_bn(F.leaky_relu(self.conv10(layer1)))
		out=out+layer1
		layer1 = self.conv11_bn(F.leaky_relu(self.conv11(out)))
		layer1 = self.conv12_bn(F.leaky_relu(self.conv12(layer1)))
		out=out+layer1
		#Dilation 2, 2 conv blocks
		out = self.conv13_bn(F.leaky_relu(self.conv13(out)))
		out = self.conv14_bn(F.leaky_relu(self.conv14(out))) 
		#Dilation 1, 2 conv blocks
		out = self.conv15_bn(F.leaky_relu(self.conv15(out)))
		out = self.conv16_bn(F.leaky_relu(self.conv16(out)))
		seg = self.seg(out)
		upsample = self.upsample(seg)
		return self.softmax(upsample)
		

class full_model(nn.Module):
	def __init__(self):
		super(full_model,self).__init__()
		self.fpn = FPN()
		self.sa = SA()
		self.globalPool = globalpool()
		self.drn = dilatedResidualNetworks()

	def forward(self, input, num_classes=7):
		ft1,ft2,ft3 = self.fpn.forward(input)
		self_attention_maps = self.sa.forward(ft1,ft2,ft3)
		globalPoolMap = self.globalPool.forward(input,self_attention_maps)
		upsampledSeg = self.drn(globalPoolMap, num_classes)
		return upsampledSeg,globalPoolMap

