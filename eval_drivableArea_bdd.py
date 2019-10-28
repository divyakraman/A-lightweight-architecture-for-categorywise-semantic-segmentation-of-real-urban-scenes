#References: https://github.com/wasidennis/AdaptSegNet/blob/master/compute_iou.py, https://github.com/jhoffman/cycada_release/blob/master/scripts/eval_fcn.py

import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
import torch
import torch.nn.functional as F
from drivableArea_model import *
import torch.optim as optim
import os



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

torch.cuda.empty_cache() #clear cached memory

#dtype = torch.FloatTensor #CPU
dtype = torch.cuda.FloatTensor #GPU; https://pytorch.org/docs/stable/tensors.html

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


batch_size = 1
iterations = len(val_images_list)
num_classes = 7 #6

'''
#6 classes
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



def fast_hist(a,b,n):
	k = (a>=0) & (a<n)
	return np.bincount(n*a[k].astype(int)+b[k], minlength=n**2).reshape(n,n)

def per_class_iu(hist):
	return np.diag(hist)/(hist.sum(1)+hist.sum(0)-np.diag(hist))


net = torch.load('models/drivableArea_bdd_7classes_15epochs.pth')


hist = np.zeros((num_classes,num_classes))
#for iteration in range(iterations):
for iteration in range(len(val_labels_list)):
	images, label_ss = get_data(iteration)
	images = images.type(dtype)
	label_ss = label_ss.type(dtype)
	pred_labels_upsampled, globalPoolMap = net(images)
	pred_labels_upsampled = pred_labels_upsampled.detach()
	pred_labels_upsampled = pred_labels_upsampled.cpu()
	pred_labels_upsampled = pred_labels_upsampled.numpy()
	del globalPoolMap
	#prediction_labels = np.zeros((1024,2048))
	pred_labels_upsampled = pred_labels_upsampled[0,:,:,:]
	pred_labels_upsampled = np.argmax(pred_labels_upsampled,0)
	label_ss = label_ss.cpu()
	label_ss = label_ss.numpy()
	label_ss = label_ss[0,:,:]
	accuracy = sum(sum(label_ss==pred_labels_upsampled))#/(720*1280)
	print(accuracy)
	pred_color_labels = np.zeros((720,1280,3))
	for i in range(len(colors)):
		pred_color_labels[np.where(pred_labels_upsampled==i)]=colors[i]
	image_name = str(iteration)+'.jpg'
	pred_color_labels = pred_color_labels/255.0
	#plt.imsave(image_name,pred_color_labels)
	hist += fast_hist(label_ss.flatten(), pred_labels_upsampled.flatten(), num_classes)
	torch.cuda.empty_cache() #clear cached memory
	print(iteration)

mIoUs = per_class_iu(hist)


for ind_class in range(num_classes):
	print('===> Class '+str(ind_class)+':\t'+str(round(mIoUs[ind_class] * 100, 2)))

print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))


print('===> Accuracy Overall: ' + str(np.diag(hist).sum() / hist.sum() * 100))
acc_percls = np.diag(hist) / (hist.sum(1) + 1e-8) 

for ind_class in range(num_classes):
	print('===> Class '+str(ind_class)+':\t'+str(round(acc_percls[ind_class] * 100, 2))) 






