import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
import torch
import torch.nn.functional as F
from model import *
import torch.optim as optim
import os



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#dtype = torch.FloatTensor #CPU
dtype = torch.cuda.FloatTensor #GPU



train_folder_imgs = '/media/data2/ee15b085/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train'
val_folder_imgs = '/media/data2/ee15b085/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val'
test_folder_imgs = '/media/data2/ee15b085/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/folder'


train_folder_labels = '/media/data2/ee15b085/cityscapes/gtFine_trainvaltest/gtFine/train'
val_folder_labels = '/media/data2/ee15b085/cityscapes/gtFine_trainvaltest/gtFine/val'
test_folder_labels = '/media/data2/ee15b085/cityscapes/gtFine_trainvaltest/gtFine/test'

train_labels_list = glob.glob(train_folder_labels+"/**/*_color.png")
val_labels_list = glob.glob(val_folder_labels+"/**/*_color.png")
train_images_list = glob.glob(train_folder_imgs+"/**/*.png")
val_images_list = glob.glob(val_folder_imgs+"/**/*.png")

train_labels_list.sort()
val_labels_list.sort()
train_images_list.sort()
val_images_list.sort()

batch_size = 18 #18 crops per image
iterations = len(train_images_list)
epochs = 1
num_classes = 20 #19+1; 1 for unknown

learning_rate = 2.5e-4
momentum = 0.9
power = 0.9
weight_decay = 0.0005

colors = [ [0,0,0],
[128,64,128],
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


def get_data(iter_num):
	images = torch.zeros([batch_size,3,327,327])
	images = images.cuda()
	#labels = torch.zeros([batch_size,num_classes,327,327])
	labels = torch.zeros([batch_size,327,327])
	labels = labels.cuda()
	#RGB_image = imageio.imread(train_images_list[int(iter_num/3)]) #1024*2048*3
	RGB_image = imageio.imread(train_images_list[int(iter_num)]) #1024*2048*3
	RGB_image = RGB_image/255.0
	RGB_image = torch.from_numpy(RGB_image)
	RGB_image = RGB_image.cuda()
	RGB_image = RGB_image.permute(2,0,1)#3*1024*2048
	#label_image = imageio.imread(train_labels_list[int(iter_num/3)]) #1024*2048*3
	label_image = imageio.imread(train_labels_list[int(iter_num)]) #1024*2048*3
	label_image = label_image[:1024,:2048,0:3]
	#make labels compatible for semantic segmentation
	#label_ss = np.zeros((1024,2048,20)) #number of classes + 1 for unknown class
	label_ss = np.zeros((1024,2048))
	#np.where,np.argwhere,np.nonzero
	for i in range(num_classes):
		#temp = np.zeros((1024,2048))
		label_ss[np.where((label_image == colors[i]).all(axis=2))] = i
		#label_ss[:,:,i] = temp #Pixels with class belonging to ith class marked by 1; useful for cross entropy loss
	#del temp
	label_ss = torch.from_numpy(label_ss)
	label_ss = label_ss.cuda()
	#label_ss = label_ss.permute(2,0,1)
	j = 0
	for i in range(6):
		images[j,:,:,:] = RGB_image[:,0:327,327*i:327*(i+1)]
		labels[j,:,:] = label_ss[0:327,327*i:327*(i+1)]
		j = j+1
	for i in range(6):
		images[j,:,:,:] = RGB_image[:,327:654,327*i:327*(i+1)]
		labels[j,:,:] = label_ss[327:654,327*i:327*(i+1)]
		j = j+1
	for i in range(6):
		images[j,:,:,:] = RGB_image[:,654:981,327*i:327*(i+1)]
		labels[j,:,:] = label_ss[654:981,327*i:327*(i+1)]
		j = j+1
	'''
	if(iter_num%3==0):
		for i in range(batch_size):
			images[i,:,:,:] = RGB_image[:,0:327,327*i:327*(i+1)]
			labels[i,:,:] = label_ss[0:327,327*i:327*(i+1)]
		
		images[0,:,:,:] = RGB_image[:,0:327,0:327]
		images[1,:,:,:] = RGB_image[:,0:327,327:654]
		images[2,:,:,:] = RGB_image[:,0:327,654:981]
		images[3,:,:,:] = RGB_image[:,0:327,981:1308]
		images[4,:,:,:] = RGB_image[:,0:327,1308:1635]
		images[5,:,:,:] = RGB_image[:,0:327,1635:1962]
		labels[0,:,:] = label_ss[0:327,0:327]
		labels[1,:,:] = label_ss[0:327,327:654]
		labels[2,:,:] = label_ss[0:327,654:981]
		labels[3,:,:] = label_ss[0:327,981:1308]
		labels[4,:,:] = label_ss[0:327,1308:1635]
		labels[5,:,:] = label_ss[0:327,1635:1962]
		
	elif(iter_num%3==1):
		for i in range(batch_size):
			images[i,:,:,:] = RGB_image[:,327:654,327*i:327*(i+1)]
			labels[i,:,:] = label_ss[327:654,327*i:327*(i+1)]
		
		images[0,:,:,:] = RGB_image[:,327:654,0:327]
		images[1,:,:,:] = RGB_image[:,327:654,327:654]
		images[2,:,:,:] = RGB_image[:,327:654,654:981]
		images[3,:,:,:] = RGB_image[:,327:654,981:1308]
		images[4,:,:,:] = RGB_image[:,327:654,1308:1635]
		images[5,:,:,:] = RGB_image[:,327:654,1635:1962]
		labels[0,:,:] = label_ss[327:654,0:327]
		labels[1,:,:] = label_ss[327:654,327:654]
		labels[2,:,:] = label_ss[327:654,654:981]
		labels[3,:,:] = label_ss[327:654,981:1308]
		labels[4,:,:] = label_ss[327:654,1308:1635]
		labels[5,:,:] = label_ss[327:654,1635:1962]
		
	#if(iter_num%3==2):
	else:
		for i in range(batch_size):
			images[i,:,:,:] = RGB_image[:,654:981,327*i:327*(i+1)]
			labels[i,:,:] = label_ss[654:981,327*i:327*(i+1)]
		
		images[0,:,:,:] = RGB_image[:,654:981,0:327]
		images[1,:,:,:] = RGB_image[:,654:981,327:654]
		images[2,:,:,:] = RGB_image[:,654:981,654:981]
		images[3,:,:,:] = RGB_image[:,654:981,981:1308]
		images[4,:,:,:] = RGB_image[:,654:981,1308:1635]
		images[5,:,:,:] = RGB_image[:,654:981,1635:1962]
		labels[0,:,:] = label_ss[654:981,0:327]
		labels[1,:,:] = label_ss[654:981,327:654]
		labels[2,:,:] = label_ss[654:981,654:981]
		labels[3,:,:] = label_ss[654:981,981:1308]
		labels[4,:,:] = label_ss[654:981,1308:1635]
		labels[5,:,:] = label_ss[654:981,1635:1962]
	'''

	del RGB_image, label_ss
	return images,labels


#net = full_model()
#net = net.cuda()
net = torch.load('models/segmentation_cityscapes.pth')
parameters = net.parameters()

optimizer = optim.SGD(parameters, lr = learning_rate, momentum = momentum, weight_decay = weight_decay)
optimizer.zero_grad()
loss_fn = torch.nn.CrossEntropyLoss()


for epoch in range(epochs):
	loss = 0
	accum_loss = 0
	optimizer.zero_grad()
	new_lr = learning_rate * ((1 - float(epoch) / epochs) ** (power)) # Find new learning rate
	for param_group in optimizer.param_groups:
		param_group['lr'] = new_lr  # Update new learning rate
	for iteration in range(iterations):
		images, label_ss = get_data(iteration)
		images = images.type(dtype)
		label_ss = label_ss.type(dtype)
		pred_labels_upsampled, globalPoolMap = net(images)
		#print("Loading done")
		#pred_labels_upsampled = pred_labels_upsampled.long()
		label_ss = label_ss.long()
		loss = loss_fn(pred_labels_upsampled, label_ss)
		loss = loss/iterations
		accum_loss = accum_loss+loss
		#print(accum_loss)
		loss.backward()
		if(iteration%1==0):
			print(iteration)
	optimizer.step()
	print("Epoch ",epoch," loss is ",accum_loss)


torch.save(net, 'models/segmentation_cityscapes.pth')




