import numpy as np 
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from GCN_BR import GCN,BR 
#import matplotlib.pyplot as plt  
#import scipy.misc as smisc
#import random


class FPN(nn.Module):
	def __init__(self):
		 super(FPN, self).__init__()
		 self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv4 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.down1 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv1_bn = nn.BatchNorm2d(32)
		 self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv5 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.down2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv2_bn = nn.BatchNorm2d(64)
		 self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv6 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.down3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=4,stride=2,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv3_bn = nn.BatchNorm2d(128)
		 self.lateral3 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.lateral2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.lateral1 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.upsample3 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
		 self.upsample2 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
		 self.conv7 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv8 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv9 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.newft3_bn = nn.BatchNorm2d(64)
		 self.newft2_bn = nn.BatchNorm2d(64)
		 self.newft1_bn = nn.BatchNorm2d(64)



	def forward(self,input):
		ft1 = self.conv1_bn(F.leaky_relu(self.down1(F.leaky_relu(self.conv4(F.leaky_relu(self.conv1(input))))))) #1*32*160*160
		ft2 = self.conv2_bn(F.leaky_relu(self.down2(F.leaky_relu(self.conv5(F.leaky_relu(self.conv2(ft1)))))))  #1*64*80*80
		ft3 = self.conv3_bn(F.leaky_relu(self.down3(F.leaky_relu(self.conv6(F.leaky_relu(self.conv3(ft2)))))))  #1*128*40*40  
		new_ft3 = self.newft3_bn(F.leaky_relu(self.conv7(self.lateral3(ft3)))) #1*64*40*40 ; name as lateral3
		new_ft2 = self.newft2_bn(F.leaky_relu(self.conv8(self.lateral2(ft2)+self.upsample3(new_ft3)))) #1*64*80*80
		new_ft1 = self.newft1_bn(F.leaky_relu(self.conv9(self.lateral1(ft1)+self.upsample2(new_ft2)))) #1*64*160*160
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


		
class full_model(nn.Module):
	def __init__(self):
		super(full_model,self).__init__()
		self.fpn = FPN()
		self.sa = SA()
		self.final_conv = nn.Conv2d(in_channels=128,out_channels=7,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.softmax = nn.LogSoftmax()

	def forward(self, input, num_classes=7):
		ft1,ft2,ft3 = self.fpn.forward(input)
		sa_maps = self.sa.forward(ft1,ft2,ft3)
		out = F.upsample(self.final_conv(sa_maps), input.size()[2:], mode='bilinear', align_corners=True)
		return self.softmax(out)