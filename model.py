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
		 self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv1_bn = nn.BatchNorm2d(32)
		 self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv2_bn = nn.BatchNorm2d(64)
		 self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv3_bn = nn.BatchNorm2d(128)
		 self.lateral3 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.lateral2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.lateral1 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.upsample3 = nn.Upsample(size=(81,81),mode='bilinear',align_corners=True)
		 self.upsample2 = nn.Upsample(size=(163,163),mode='bilinear',align_corners=True)
		 self.newft3_bn = nn.BatchNorm2d(64)
		 self.newft2_bn = nn.BatchNorm2d(64)
		 self.newft1_bn = nn.BatchNorm2d(64)



	def forward(self,input):
		#input is 1*3*327*327
		'''
		Long version of the code
		ft1 = self.conv1_bn(F.relu(self.conv1(input))) #1*32*163*163
		ft2 = self.conv2_bn(F.relu(self.conv2(ft1)))  #1*64*81*81
		ft3 = self.conv3_bn(F.relu(self.conv1(ft3)))  #1*128*40*40  
		new_ft3 = self.lateral3(ft3) #1*64*40*40 ; name as lateral3
		lateral2 = self.lateral2(ft2) #1*64*81*81
		upsample3 = self.upsample3(lateral3)#1*64*81*81
		new_ft2 = self.conv2_bn(lateral2+upsample3)#1*64*81*81
		lateral1 = self.lateral1(ft1) #1*64*163*163
		upsample2 = self.upsample2(new_ft2)#1*64*163*163
		new_ft1 = self.conv2_bn(lateral1+upsample2)#1*64*163*163
		'''
		#Short code
		ft1 = self.conv1_bn(F.relu(self.conv1(input))) #1*32*163*163
		ft2 = self.conv2_bn(F.relu(self.conv2(ft1)))  #1*64*81*81
		ft3 = self.conv3_bn(F.relu(self.conv3(ft2)))  #1*128*40*40  
		new_ft3 = self.newft3_bn(self.lateral3(ft3)) #1*64*40*40 ; name as lateral3
		new_ft2 = self.newft2_bn(self.conv2_bn(self.lateral2(ft2)+self.upsample3(new_ft3))) #1*64*81*81
		new_ft1 = self.newft1_bn(self.conv2_bn(self.lateral1(ft1)+self.upsample2(new_ft2))) #1*64*163*163
		return new_ft3,new_ft2,new_ft1



class SA(nn.Module):
	def __init__(self):
		 super(SA, self).__init__()
		 #f,g,h are key value query maps for self attention. Self attention maps for each level of feature pyramid generated followed by upsampling 
		 #to 327*327 which is the original image size.
		 self.f3 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.g3 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.h3 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.f2 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.g2 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.h2 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.f1 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.g1 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.h1 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.downsample = nn.Conv2d(in_channels=96,out_channels=128,kernel_size=3,stride=2,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv1 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.conv4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		 self.upsample = nn.Upsample(size=(163,163),mode='bilinear',align_corners=True)
		 self.bn1 = nn.BatchNorm2d(32)
		 self.bn2 = nn.BatchNorm2d(32)
		 self.bn3 = nn.BatchNorm2d(32)
		 self.bn_downsample = nn.BatchNorm2d(128)
		 self.bn_conv1 = nn.BatchNorm2d(128)
		 self.bn_conv2 = nn.BatchNorm2d(128)
		 self.bn_conv3 = nn.BatchNorm2d(128)
		 self.bn_conv4 = nn.BatchNorm2d(128)


	def matrixmul(self,a,b):
		#a and b are 4 dimensional tensors batchsize*channels*height*width
		out_shape = a.size()
		output = torch.zeros(out_shape)
		for i in range(out_shape[0]):
				output[i,:,:,:]=torch.bmm(a[i,:,:,:],b[i,:,:,:])
		return output
		 

	def forward(self,new_ft1,new_ft2,new_ft3):
		sa3 = self.matrixmul(F.softmax(self.matrixmul((self.f3(new_ft3)).permute(0,1,3,2),self.g3(new_ft3)),dim=1),self.h3(new_ft3))
		sa3 = self.bn3(sa3)
		sa3 = self.upsample(sa3)
		sa2 = self.matrixmul(F.softmax(self.matrixmul((self.f2(new_ft2)).permute(0,1,3,2),self.g2(new_ft2)),dim=1),self.h2(new_ft2))
		sa2 = self.upsample(sa2)
		sa2 = self.bn2(sa2)
		sa1 = self.matrixmul(F.softmax(self.matrixmul((self.f1(new_ft1)).permute(0,1,3,2),self.g1(new_ft1)),dim=1),self.h1(new_ft1))
		sa1 = self.bn3(sa1)
		sa1 = self.upsample(sa1)
		#All final self attention maps from all levels of feature pyramid of spatial resolution 327*327
		concat = torch.cat((sa1,sa2,sa3),dim=1)	#1*96*81*81	#Difference between torch.stack and torch.cat
		downsample = self.bn_downsample(F.relu(self.downsample(concat)))#1*128*81*81
		conv = self.bn_conv1(F.relu(self.conv1(downsample))) #1*128*81*81
		conv = self.bn_conv2(F.relu(self.conv2(conv))) #1*128*81*81
		downsample = conv+downsample #residual connection; 1*128*81*81
		conv = self.bn_conv3(F.relu(self.conv3(downsample))) #1*128*81*81
		conv = self.bn_conv4(F.relu(self.conv4(conv))) #1*128*81*81
		out = conv+downsample #residual connection; 1*128*81*81
		#out is object_sa_maps
		return out

class globalpool(nn.Module):
	def __init__(self):
		super(globalpool, self).__init__()    #change output channels
		self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv1_bn = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv2_bn = nn.BatchNorm2d(16)
		self.conv3 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv3_bn = nn.BatchNorm2d(16)
		self.downsample1 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.downsample1_bn = nn.BatchNorm2d(32) #1*32*163*163
		self.conv4 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv5 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.conv4_bn = nn.BatchNorm2d(32)
		self.conv5_bn = nn.BatchNorm2d(32)
		self.downsample2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
		self.downsample2_bn = nn.BatchNorm2d(64) #1*64*81*81
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
		layer1 = self.conv1_bn(F.relu(self.conv1(input)))
		global_maps = self.conv2_bn(F.relu(self.conv2(layer1))) #1*16*327*327
		global_maps = self.conv3_bn(F.relu(self.conv3(global_maps))) #1*16*327*327
		global_maps = global_maps + layer1 #1*16*327*327
		global_maps = self.downsample1_bn(F.relu(self.downsample1(global_maps)))#1*32*163*163
		layer1 = self.conv4_bn(F.relu(self.conv4(global_maps)))
		layer1 = self.conv5_bn(F.relu(self.conv5(global_maps)))
		global_maps = global_maps+layer1
		global_maps = self.downsample2_bn(F.relu(self.downsample2(global_maps)))#1*64*81*81
		layer1 = self.conv6_bn(F.relu(self.conv6(global_maps)))
		layer1 = self.conv7_bn(F.relu(self.conv7(global_maps)))
		global_maps = global_maps+layer1
		layer1 = self.conv8_bn(F.relu(self.conv8(global_maps)))
		layer1 = self.conv9_bn(F.relu(self.conv9(global_maps)))
		global_maps = global_maps+layer1
		out = torch.cat((global_maps,object_sa_maps),dim=1)  #1*192*81*81
		return out


class dilatedResidualNetworks(nn.Module):
	def __init__(self,num_classes=30):
		super(dilatedResidualNetworks,self).__init__()
		self.downsample1 = nn.Conv2d(in_channels=192,out_channels=256,kernel_size=3,stride=2,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
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
		self.upsample = nn.Upsample(size=(327,327),mode='bilinear',align_corners=True)




	def forward(self,input,num_classes):
		#input is global and self attention concatenated feature maps
		out = self.downsample1_bn(F.relu(self.downsample1(input))) #1*256*40*40
		#Dilation 1, 2 residual blocks
		layer1 = self.conv1_bn(F.relu(self.conv1(out))) 
		layer1 = self.conv2_bn(F.relu(self.conv2(layer1))) 
		out=out+layer1
		layer1 = self.conv3_bn(F.relu(self.conv3(out))) 
		layer1 = self.conv4_bn(F.relu(self.conv4(layer1))) 
		out=out+layer1
		#Dilation 2, 2 residual blocks
		layer1 = self.conv5_bn(F.relu(self.conv5(out))) 
		layer1 = self.conv6_bn(F.relu(self.conv6(layer1))) 
		out=out+layer1
		layer1 = self.conv7_bn(F.relu(self.conv7(out))) 
		layer1 = self.conv8_bn(F.relu(self.conv8(layer1))) 
		out=out+layer1
		#Let's go deeper
		out = self.deeper_bn(F.relu(self.deeper(out))) 
		#Dilation 4, 2 residual blocks
		layer1 = self.conv9_bn(F.relu(self.conv9(out))) 
		layer1 = self.conv10_bn(F.relu(self.conv10(layer1))) 
		out=out+layer1
		layer1 = self.conv11_bn(F.relu(self.conv11(out))) 
		layer1 = self.conv12_bn(F.relu(self.conv12(layer1))) 
		out=out+layer1
		#Dilation 2, 2 conv blocks
		out = self.conv13_bn(F.relu(self.conv13(out))) 
		out = self.conv14_bn(F.relu(self.conv14(out))) 
		#Dilation 1, 2 conv blocks
		out = self.conv15_bn(F.relu(self.conv15(out))) 
		out = self.conv16_bn(F.relu(self.conv16(out))) 
		seg = self.seg(out)
		upsample = self.upsample(seg)
		return self.softmax(upsample),seg
		

class full_model(nn.Module):
	def __init__(self):
		super(full_model,self).__init__()
		self.fpn = FPN()
		self.sa = SA()
		self.globalPool = globalpool()
		self.drn = dilatedResidualNetworks()

	def forward(self, input, num_classes=30):
		ft1,ft2,ft3 = self.fpn.forward(input)
		self_attention_maps = self.sa.forward(ft1,ft2,ft3)
		globalPool = self.globalPool.forward(input,self_attention_maps)
		upsampledSeg,seg = self.dilatedResidualNetworks(globalPool, num_classes)
		return upsampledSeg,seg