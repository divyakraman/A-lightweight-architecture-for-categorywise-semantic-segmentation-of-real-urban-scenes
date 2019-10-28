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
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#dtype = torch.FloatTensor #CPU
dtype = torch.cuda.FloatTensor #GPU


#train_folder_imgs = '/media/data3/Athira/Works/MyWorks/Myworks/AdaptSegNet/data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train'
train_folder_imgs = '/media/data2/ee15b085/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train'
val_folder_imgs = '/media/data2/ee15b085/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val'
test_folder_imgs = '/media/data2/ee15b085/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/folder'


#train_folder_labels = '/media/data3/Athira/Works/MyWorks/Myworks/AdaptSegNet/data/cityscapes/gtFine_trainvaltest/gtFine/train'
train_folder_labels = '/media/data2/ee15b085/cityscapes/gtFine_trainvaltest/gtFine/train'
val_folder_labels = '/media/data2/ee15b085/cityscapes/gtFine_trainvaltest/gtFine/val'
test_folder_labels = '/media/data2/ee15b085/cityscapes/gtFine_trainvaltest/gtFine/test'

train_labels_list = glob.glob(train_folder_labels+"/**/*_labelIds.png")
val_labels_list = glob.glob(val_folder_labels+"/**/*_labelIds.png")
train_images_list = glob.glob(train_folder_imgs+"/**/*.png")
val_images_list = glob.glob(val_folder_imgs+"/**/*.png")

train_labels_list.sort()
val_labels_list.sort()
train_images_list.sort()
val_images_list.sort()

batch_size = 1 #train one image at a time, entire image
step_size = 1
iterations = len(train_images_list)
epochs = 30
#num_classes = 6
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
	images = torch.zeros([batch_size,3,1024,2048])
	images = images.cuda()
	RGB_image = imageio.imread(train_images_list[int(iter_num)]) #1024*2048*3
	RGB_image = RGB_image/255.0
	RGB_image = torch.from_numpy(RGB_image)
	RGB_image = RGB_image.cuda()
	RGB_image = RGB_image.permute(2,0,1)#3*1024*2048
	images[0,:,:,:] = RGB_image
	label_image = imageio.imread(train_labels_list[int(iter_num)]) #1024*2048
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
	labels[0,:,:] = label_image
	
	del RGB_image,label_image
	return images,labels


#net = full_model()
#net = net.cuda()
net = torch.load('models/drivableArea_cityscapes_7classes_15epochs.pth')
'''
for param in net.parameters():
	torch.nn.init.normal_(param,mean=0.001,std=0.001)
'''
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


for epoch in range(15,epochs):
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
		accuracy = sum(sum(sum(label_ss==prediction_labels)))/(1024*2048)
		print("Epoch ",epoch, " Iteration ",iteration, " Accuracy is ",accuracy)
		hist = hist + fast_hist(label_ss.flatten(), prediction_labels.flatten(), num_classes)
		mIoUs = per_class_iu(hist)
		print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
		torch.cuda.empty_cache() #clear cached memory
		print(np.sum(prediction_labels),np.sum(label_ss))
		if(iteration%100==0):
			torch.save(net, 'models/drivableArea_cityscapes_7classes_30epochs.pth')





