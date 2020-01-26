import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import math

#https://github.com/SConsul/Global_Convolutional_Network/blob/master/model/GCN.py
class GCN(nn.Module):
    def __init__(self,c,out_c,k=7): #out_Channel=21 in paper
        super(GCN, self).__init__()
        self.padding1 = int((k-1)/2)
        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k,1), padding = (self.padding1,0))
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1,k), padding =(0,self.padding1))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1,k), padding =(self.padding1,0))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k,1), padding =(0,self.padding1))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x

#https://github.com/SConsul/Global_Convolutional_Network/blob/master/model/BR.py
class BR(nn.Module):
    def __init__(self, out_c):
        super(BR, self).__init__()
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)

    def forward(self,x):
        x_res = x
        x_res = self.conv1(x_res)
        x_res = self.relu(x_res)
        x_res = self.conv2(x_res)
        x = x + x_res
        return x