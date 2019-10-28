import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
import torch
import torch.nn.functional as F
from drivableArea_deconv_model import *
import torch.optim as optim
import os



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#dtype = torch.FloatTensor #CPU
dtype = torch.cuda.FloatTensor #GPU

train_folder_imgs = '/media/data2/ee15b085/BerkeleyDeepDrive/bdd100k_seg/bdd100k/seg/images/train'
val_folder_imgs = '/media/data2/ee15b085/BerkeleyDeepDrive/bdd100k_seg/bdd100k/seg/images/val'

train_folder_labels = '/media/data2/ee15b085/BerkeleyDeepDrive/bdd100k_seg/bdd100k/seg/labels/train'
val_folder_labels = '/media/data2/ee15b085/BerkeleyDeepDrive/bdd100k_seg/bdd100k/seg/labels/val'

train_labels_list = glob.glob(train_folder_labels+"/*.png")
val_labels_list = glob.glob(val_folder_labels+"/*.png")
train_images_list = glob.glob(train_folder_imgs+"/*.jpg") #7000 images
val_images_list = glob.glob(val_folder_imgs+"/*.jpg") #1000 images

train_labels_list.sort()
val_labels_list.sort()
train_images_list.sort()
val_images_list.sort()

batch_size = 1 #train one image at a time, entire image
step_size = 1
iterations = len(train_images_list)
epochs = 20
num_classes = 7

learning_rate = 1e-3
momentum = 0.9
power = 0.9
weight_decay = 0.0005

'''
6 classes
colors = [ [128,64,128], 
[244,35,232], 
[70,70,70], 
[107,142,35], 
[70,130,180], 
[220,20,60], 
]
'''
# 7 classes
colors = [[128,64,128],
[70,70,70],
[153,153,153],
[107,142,35],
[70,130,180],
[220,20,60],
[0,0,142]]


def get_data(iter_num):
	#images = torch.zeros([batch_size,3,327,327])
	images = torch.zeros([batch_size,3,720,1280])
	images = images.cuda()
	#labels = torch.zeros([batch_size,327,327])
	labels = torch.zeros([batch_size,720,1280])
	labels = labels.cuda()
	RGB_image = imageio.imread(train_images_list[int(iter_num)]) #720*1280*3
	RGB_image = RGB_image/255.0
	RGB_image = torch.from_numpy(RGB_image)
	RGB_image = RGB_image.cuda()
	RGB_image = RGB_image.permute(2,0,1)#3*720*1280
	images[0,:,:,:] = RGB_image
	del RGB_image
	label_image = imageio.imread(train_labels_list[int(iter_num)])
	label_image = np.array(label_image) 
	label_image[np.where(label_image == 0)] = 0 #road 0
	label_image[np.where(label_image == 1)] = 0 #sidewalk 1
	label_image[np.where(label_image == 2)] = 1 #building 2
	label_image[np.where(label_image == 3)] = 1 #wall 3
	label_image[np.where(label_image == 4)] = 1 #fence 4
	label_image[np.where(label_image == 5)] = 2 #pole 5
	label_image[np.where(label_image == 6)] = 2 #traffic light 6
	label_image[np.where(label_image == 7)] = 2 #traffic sign 7
	label_image[np.where(label_image == 8)] = 3 #vegetation 8
	label_image[np.where(label_image == 9)] = 3 #nature 9
	label_image[np.where(label_image == 10)] = 4 #sky 10
	label_image[np.where(label_image == 11)] = 5 #person 11
	label_image[np.where(label_image == 12)] = 5 #rider 12
	label_image[np.where(label_image == 13)] = 6 #car 13
	label_image[np.where(label_image == 14)] = 6 #truck 14
	label_image[np.where(label_image == 15)] = 6 #bus 15
	label_image[np.where(label_image == 16)] = 6 #train 16
	label_image[np.where(label_image == 17)] = 6 #motorcycle 17
	label_image[np.where(label_image == 18)] = 6 #bicycle 18
	label_image[np.where(label_image == -1)] = 255
	label_image = torch.from_numpy(label_image)
	label_image = label_image.cuda()
	labels[0,:,:] = label_image
	del label_image

	return images,labels

#net = full_model()
#net = net.cuda()
net = torch.load('models/drivable_deconv_bdd_10epochs.pth')

parameters = net.parameters()

optimizer = optim.SGD(parameters, lr = learning_rate, momentum = momentum, weight_decay = weight_decay)
optimizer.zero_grad()
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)  #https://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#CrossEntropyLoss
 
hist = np.zeros((num_classes,num_classes))
def fast_hist(a,b,n):
	k = (a>=0) & (a<n)
	return np.bincount(n*a[k].astype(int)+b[k], minlength=n**2).reshape(n,n)

def per_class_iu(hist):
	return np.diag(hist)/(hist.sum(1)+hist.sum(0)-np.diag(hist))


for epoch in range(10,epochs):
	hist = np.zeros((num_classes,num_classes))
	loss = 0
	new_lr = learning_rate * ((1 - float(epoch) / epochs) ** (power)) # Find new learning rate
	for param_group in optimizer.param_groups:
		param_group['lr'] = new_lr  # Update new learning rate
	for iteration in range(iterations):
		images, label_ss = get_data(iteration)
		images = images.type(dtype)
		label_ss = label_ss.type(dtype)
		pred_labels_upsampled, globalPoolMap = net(images)
		label_ss = label_ss.long()
		loss = loss_fn(pred_labels_upsampled, label_ss) 
		loss = loss/step_size
		loss.backward()
		if(iteration%step_size==0):
			optimizer.step()
			optimizer.zero_grad()
		pred_labels_upsampled = pred_labels_upsampled.detach()
		pred_labels_upsampled = pred_labels_upsampled.cpu()
		pred_labels_upsampled = pred_labels_upsampled.numpy()
		prediction_labels = np.argmax(pred_labels_upsampled,1)
		label_ss = label_ss.cpu()
		label_ss = label_ss.numpy()
		accuracy = sum(sum(sum(label_ss==prediction_labels)))/(720*1280)
		print("Epoch ",epoch, " Iteration ",iteration, " Accuracy is ",accuracy)
		hist = hist + fast_hist(label_ss.flatten(), prediction_labels.flatten(), num_classes)
		mIoUs = per_class_iu(hist)
		print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
		torch.cuda.empty_cache() #clear cached memory
		print(np.sum(prediction_labels),np.sum(label_ss))
		if(iteration%100==0):
			torch.save(net, 'models/drivable_deconv_bdd_20epochs.pth')
			mIoUs = per_class_iu(hist)
			for ind_class in range(num_classes):
				print('===> Class '+str(ind_class)+':\t'+str(round(mIoUs[ind_class] * 100, 2)))




