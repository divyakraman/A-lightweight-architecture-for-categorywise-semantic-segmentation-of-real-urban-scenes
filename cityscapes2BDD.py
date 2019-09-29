'''
References: https://github.com/wasidennis/AdaptSegNet/blob/master/train_gta2cityscapes_multi.py

'''

import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
import torch
import torch.nn.functional as F
from model import *
from discriminator import *
import torch.optim as optim
import os



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#dtype = torch.FloatTensor #CPU
dtype = torch.cuda.FloatTensor #GPU



train_folder_imgs_cityscapes = '/media/data2/ee15b085/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train'
val_folder_imgs_cityscapes = '/media/data2/ee15b085/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val'
#test_folder_imgs_cityscapes = '/media/data2/ee15b085/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/folder'


train_folder_labels_cityscapes = '/media/data2/ee15b085/cityscapes/gtFine_trainvaltest/gtFine/train'
val_folder_labels_cityscapes = '/media/data2/ee15b085/cityscapes/gtFine_trainvaltest/gtFine/val'
#test_folder_labels_cityscapes = '/media/data2/ee15b085/cityscapes/gtFine_trainvaltest/gtFine/test'

train_labels_list_cityscapes = glob.glob(train_folder_labels_cityscapes+"/**/*_color.png")
val_labels_list_cityscapes = glob.glob(val_folder_labels_cityscapes+"/**/*_color.png")
train_images_list_cityscapes = glob.glob(train_folder_imgs_cityscapes+"/**/*.png") #2975 images
val_images_list_cityscapes = glob.glob(val_folder_imgs_cityscapes+"/**/*.png") #500 images

train_labels_list_cityscapes.sort()
val_labels_list_cityscapes.sort()
train_images_list_cityscapes.sort()
val_images_list_cityscapes.sort()

train_folder_imgs_BDD = '/media/data2/ee15b085/BerkeleyDeepDrive/bdd100k_seg/bdd100k/seg/images/train'
val_folder_imgs_BDD = '/media/data2/ee15b085/BerkeleyDeepDrive/bdd100k_seg/bdd100k/seg/images/val'


train_folder_labels_BDD = '/media/data2/ee15b085/BerkeleyDeepDrive/bdd100k_seg/bdd100k/seg/labels/train'
val_folder_labels_BDD = '/media/data2/ee15b085/BerkeleyDeepDrive/bdd100k_seg/bdd100k/seg/labels/val'


train_labels_list_BDD = glob.glob(train_folder_labels_BDD+"/*.png")
val_labels_list_BDD = glob.glob(val_folder_labels_BDD+"/*.png")
train_images_list_BDD = glob.glob(train_folder_imgs_BDD+"/*.jpg") #7000 images
val_images_list_BDD = glob.glob(val_folder_imgs_BDD+"/*.jpg") #1000 images

train_labels_list_BDD.sort()
val_labels_list_BDD.sort()
train_images_list_BDD.sort()
val_images_list_BDD.sort()

# 18 crops per image for cityscapes; 8 crops per image for BDD. 
batch_size_cityscapes = 6 #6 images from source cityscapes and 4 from target bdd
batch_size_bdd = 4
iterations = 3*len(train_images_list_cityscapes) #18 crops per image in cityscapes; therefore 3*len to train across all patches of all cityscapes images. (2975*3)/2 = 4462.5 images of BDD used.

epochs = 1
num_classes = 20 #19+1; 1 for unknown

learning_rate = 2.5e-4
learning_rate_D = 1e-4
momentum = 0.9
power = 0.9
weight_decay = 0.0005
lambda_D1 = 0.001
lambda_D2 = 0.0002

colors = [ [128,64,128],
[244,35,232],
[70,70,70],
[102,102,156],
[190,153,153],
[153,153,153],
[250,170,30],
[220,220,0],
[107,142,35],
[152,251,152],
[0,130,180],
[220,20,60],
[255,0,0],
[0,0,142],
[0,0,70],
[0,60,100],
[0,80,100],
[0,0,230],
[119,11,32] ] 


def get_data_cityscapes(iter_num):
	images = torch.zeros([batch_size_cityscapes,3,327,327])
	images = images.cuda()
	labels = torch.zeros([batch_size_cityscapes,327,327])
	labels = labels.cuda()
	RGB_image = imageio.imread(train_images_list_cityscapes[int(iter_num/3)]) #1024*2048*3
	RGB_image = RGB_image/255.0
	#RGB_image = imageio.imread(train_images_list_cityscapes[int(iter_num)]) #1024*2048*3
	RGB_image = torch.from_numpy(RGB_image)
	RGB_image = RGB_image.cuda()
	RGB_image = RGB_image.permute(2,0,1)#3*1024*2048
	label_image = imageio.imread(train_labels_list_cityscapes[int(iter_num/3)]) #1024*2048*3
	#label_image = imageio.imread(train_labels_list_cityscapes[int(iter_num)]) #1024*2048*3
	label_image = label_image[:1024,:2048,0:3]
	label_ss = np.ones((1024,2048))
	label_ss = label_ss*19
	for i in range(len(colors)):
		label_ss[np.where((label_image == colors[i]).all(axis=2))] = i
	label_ss = torch.from_numpy(label_ss)
	label_ss = label_ss.cuda()
	if(iter_num%3==0):
		for i in range(6):
			images[i,:,:,:] = RGB_image[:,0:327,327*i:327*(i+1)]
			labels[i,:,:] = label_ss[0:327,327*i:327*(i+1)]
	if(iter_num%3==1):
		for i in range(6):
			images[i,:,:,:] = RGB_image[:,327:654,327*i:327*(i+1)]
			labels[i,:,:] = label_ss[327:654,327*i:327*(i+1)]
	if(iter_num%3==2):
		for i in range(6):
			images[i,:,:,:] = RGB_image[:,654:981,327*i:327*(i+1)]
			labels[i,:,:] = label_ss[654:981,327*i:327*(i+1)]

	del RGB_image, label_ss
	return images,labels

def get_data_BDD(iter_num):
	images = torch.zeros([batch_size_bdd,3,327,327])
	images = images.cuda()
	labels = torch.zeros([batch_size_bdd,327,327])
	labels = labels.cuda()
	RGB_image = imageio.imread(train_images_list_BDD[int(iter_num/2)]) #720*1280*3
	RGB_image = RGB_image/255.0
	#RGB_image = imageio.imread(train_images_list_BDD[int(iter_num)]) #720*1280*3
	RGB_image = torch.from_numpy(RGB_image)
	RGB_image = RGB_image.cuda()
	RGB_image = RGB_image.permute(2,0,1)#3*720*1280
	label_ss = imageio.imread(train_labels_list_BDD[int(iter_num/2)])
	label_ss = np.array(label_ss)
	label_ss = torch.from_numpy(label_ss)
	label_ss = label_ss.cuda()
	crop_boundaries_column = np.array([0,327,327,654,654,981,953,1280])
	if(iter_num%2==0):
		for i in range(4):
			images[i,:,:,:] = RGB_image[:,0:327,crop_boundaries_column[2*i]:crop_boundaries_column[2*i+1]]
			labels[i,:,:] = label_ss[0:327,crop_boundaries_column[2*i]:crop_boundaries_column[2*i+1]]
	if(iter_num%2==1):
		for i in range(4):
			images[i,:,:,:] = RGB_image[:,393:720,crop_boundaries_column[2*i]:crop_boundaries_column[2*i+1]]
			labels[i,:,:] = label_ss[393:720,crop_boundaries_column[2*i]:crop_boundaries_column[2*i+1]]
	
	del RGB_image,label_ss
	
	return images,labels


#net = full_model()
#net = net.cuda()
net = torch.load('models/net_cityscapes2BDD1.pth')
net.train()
parameters = net.parameters()

#model_D1 = FCDiscriminator(num_classes = num_classes)  #At segmentation softmax outputs
#model_D1 = model_D1.cuda()
model_D1 = torch.load('models/D1_cityscapes2BDD1.pth')
#model_D1.train()
#model_D2 = FCDiscriminator(num_classes = 192) #From globalpool output
#model_D2 = model_D2.cuda()
model_D2 = torch.load('models/D2_cityscapes2BDD1.pth')
#model_D2.train()

optimizer = optim.SGD(parameters, lr = learning_rate, momentum = momentum, weight_decay = weight_decay)
optimizer.zero_grad()
loss_fn = torch.nn.CrossEntropyLoss()

optimizer_D1 = optim.Adam(model_D1.parameters(), lr = learning_rate_D, betas = (0.9,0.99))
optimizer_D1.zero_grad()
optimizer_D2 = optim.Adam(model_D1.parameters(), lr = learning_rate_D, betas = (0.9,0.99))
optimizer_D2.zero_grad()

bce_loss = torch.nn.BCEWithLogitsLoss()

#Labels for adversarial training
source_label = 0
target_label = 1



for epoch in range(epochs):
	loss = 0
	accum_loss = 0
	optimizer.zero_grad()
	optimizer_D1.zero_grad()
	optimizer_D2.zero_grad()

	new_lr = learning_rate * ((1 - float(epoch) / epochs) ** (power)) # Find new learning rate
	for param_group in optimizer.param_groups:
		param_group['lr'] = new_lr  # Update new learning rate
	for param_group in optimizer_D1.param_groups:
		param_group['lr'] = new_lr  # Update new learning rate
	for param_group in optimizer_D2.param_groups:
		param_group['lr'] = new_lr  # Update new learning rate


	for iteration in range(iterations):
		#Load data
		images_cs, labels_cs = get_data_cityscapes(iteration)
		images_bdd, labels_bdd = get_data_BDD(iteration)
		images_cs = images_cs.type(dtype)
		images_bdd = images_bdd.type(dtype)
		labels_cs = labels_cs.type(dtype)
		labels_cs = labels_cs.long() 
		labels_bdd = labels_bdd.type(dtype)
		labels_bdd = labels_bdd.long()

		#Train generator

		#Don't accumulate grads in D
		for param in model_D1.parameters():
			param.requires_grad = False

		for param in model_D2.parameters():
			param.requires_grad = False

		#Train with source

		pred_labels_cs, globalPoolMap_cs = net(images_cs)
		loss = loss_fn(pred_labels_cs, labels_cs)
		loss = loss/iterations
		loss.backward()

		#Train with target

		pred_labels_bdd, globalPoolMap_bdd = net(images_bdd)
		D_out1 = model_D1(F.softmax(pred_labels_bdd))
		D_out2 = model_D2(F.softmax(globalPoolMap_bdd))

		loss_adv_target1 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())
		loss_adv_target2 = bce_loss(D_out2, Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())
		loss = lambda_D1*loss_adv_target1 + lambda_D2 *loss_adv_target2
		loss = loss/iterations
		loss.backward()

		#train D

		#bring back requires_grad
		for param in model_D1.parameters():
			param.requires_grad = True

		for param in model_D2.parameters():
			param.requires_grad = True

		#train with source
		pred_labels_cs = pred_labels_cs.detach()
		globalPoolMap_cs = globalPoolMap_cs.detach()

		D_out1 = model_D1(F.softmax(pred_labels_cs))
		D_out2 = model_D2(F.softmax(globalPoolMap_cs))

		loss_D1 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())
		loss_D2 = bce_loss(D_out2, Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())
		loss_D1 = loss_D1/iterations/2
		loss_D2 = loss_D2/iterations/2
		loss_D1.backward()
		loss_D2.backward()

		#train with target
		pred_labels_bdd = pred_labels_bdd.detach()
		globalPoolMap_bdd = globalPoolMap_bdd.detach()

		D_out1 = model_D1(F.softmax(pred_labels_bdd))
		D_out2 = model_D2(F.softmax(globalPoolMap_bdd))

		loss_D1 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda())
		loss_D2 = bce_loss(D_out2, Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda())
		loss_D1 = loss_D1/iterations/2
		loss_D2 = loss_D2/iterations/2
		loss_D1.backward()
		loss_D2.backward()

		if(iteration%1==0):
			print(iteration)
	optimizer.step()
	optimizer_D1.step()
	optimizer_D2.step()
	


torch.save(net, 'models/net_cityscapes2BDD1.pth')
torch.save(model_D1, 'models/D1_cityscapes2BDD1.pth')
torch.save(model_D2, 'models/D2_cityscapes2BDD1.pth')


		






			

