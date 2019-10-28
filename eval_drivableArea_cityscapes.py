import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
import torch
import torch.nn.functional as F
from drivableArea_model import *
import torch.optim as optim
import os
from PIL import Image


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

torch.cuda.empty_cache() #clear cached memory

#dtype = torch.FloatTensor #CPU
dtype = torch.cuda.FloatTensor #GPU; https://pytorch.org/docs/stable/tensors.html

#train_folder_imgs = '/media/data2/ee15b085/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train'
val_folder_imgs = '/media/data2/ee15b085/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val'
#test_folder_imgs = '/media/data2/ee15b085/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/folder'


#train_folder_labels = '/media/data2/ee15b085/cityscapes/gtFine_trainvaltest/gtFine/train'
val_folder_labels = '/media/data2/ee15b085/cityscapes/gtFine_trainvaltest/gtFine/val'
#test_folder_labels = '/media/data2/ee15b085/cityscapes/gtFine_trainvaltest/gtFine/test'

#train_labels_list = glob.glob(train_folder_labels+"/**/*_labelIds.png")
val_labels_list = glob.glob(val_folder_labels+"/**/*_labelIds.png")
#train_images_list = glob.glob(train_folder_imgs+"/**/*.png")
val_images_list = glob.glob(val_folder_imgs+"/**/*.png")

#train_labels_list.sort()
val_labels_list.sort()
#train_images_list.sort()
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
	#images = torch.zeros([batch_size,3,512,1024])
	images = torch.zeros([batch_size,3,1024,2048])
	images = images.cuda()
	RGB_image = imageio.imread(val_images_list[int(iter_num)]) #1024*2048*3
	#RGB_image = Image.open(val_images_list[int(iter_num)]) #1024*2048*3
	#RGB_image = RGB_image.resize((1024,512), Image.BILINEAR)
	RGB_image = np.array(RGB_image) #downsample image by 2
	RGB_image = RGB_image/255.0
	RGB_image = torch.from_numpy(RGB_image)
	RGB_image = RGB_image.cuda()
	RGB_image = RGB_image.permute(2,0,1)#3*1024*2048
	images[0,:,:,:] = RGB_image
	label_image = imageio.imread(val_labels_list[int(iter_num)]) #1024*2048
	#label_image = Image.open(val_labels_list[int(iter_num)]) #1024*2048
	#label_image = label_image.resize((1024,512), Image.NEAREST)
	label_image = np.array(label_image)

	label_image[np.where(label_image == 0)] = 255
	label_image[np.where(label_image == 1)] = 255
	label_image[np.where(label_image == 2)] = 255
	label_image[np.where(label_image == 3)] = 255
	label_image[np.where(label_image == 4)] = 255
	label_image[np.where(label_image == 5)] = 255
	label_image[np.where(label_image == 6)] = 255
	label_image[np.where(label_image == 7)] = 0 #road
	label_image[np.where(label_image == 8)] = 0 #sidewalk
	label_image[np.where(label_image == 9)] = 255
	label_image[np.where(label_image == 10)] = 255
	label_image[np.where(label_image == 11)] = 1 #building
	label_image[np.where(label_image == 12)] = 1 #wall
	label_image[np.where(label_image == 13)] =  1 #fence
	label_image[np.where(label_image == 14)] = 255
	label_image[np.where(label_image == 15)] = 255
	label_image[np.where(label_image == 16)] = 255
	label_image[np.where(label_image == 17)] = 2 #pole
	label_image[np.where(label_image == 18)] = 255
	label_image[np.where(label_image == 19)] = 2 #traffic light
	label_image[np.where(label_image == 20)] = 2 #traffic sign
	label_image[np.where(label_image == 21)] = 3 #vegetation
	label_image[np.where(label_image == 22)] = 3 #nature
	label_image[np.where(label_image == 23)] = 4 #sky
	label_image[np.where(label_image == 24)] = 5 #person
	label_image[np.where(label_image == 25)] = 5 #rider
	label_image[np.where(label_image == 26)] = 6 #car
	label_image[np.where(label_image == 27)] = 6 #truck
	label_image[np.where(label_image == 28)] = 6 #bus
	label_image[np.where(label_image == 29)] = 255
	label_image[np.where(label_image == 30)] = 255
	label_image[np.where(label_image == 31)] = 6 #train
	label_image[np.where(label_image == 32)] = 6 #motorcycle
	label_image[np.where(label_image ==33)] = 6 #bicycle
	label_image[np.where(label_image == -1)] = 255

	label_image = torch.from_numpy(label_image)
	label_image = label_image.cuda()
	labels = torch.zeros([batch_size,1024,2048])
	#labels = torch.zeros([batch_size,512,1024])
	labels[0,:,:] = label_image
	
	del RGB_image,label_image
	return images,labels

def fast_hist(a,b,n):
	k = (a>=0) & (a<n)
	return np.bincount(n*a[k].astype(int)+b[k], minlength=n**2).reshape(n,n)

def per_class_iu(hist):
	return np.diag(hist)/(hist.sum(1)+hist.sum(0)-np.diag(hist))

#net = torch.load('models/drivableArea_cityscapes.pth')
net = torch.load('models/drivableArea_cityscapes_7classes_30epochs.pth')

hist = np.zeros((num_classes,num_classes))
#for iteration in range(5):
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
	pred_color_labels = np.zeros((1024,2048,3))
	#pred_color_labels = np.zeros((512,1024,3))
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






