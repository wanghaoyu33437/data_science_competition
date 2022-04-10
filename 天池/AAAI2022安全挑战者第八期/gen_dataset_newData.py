import numpy as np
import torchvision
import random
from Bag_trick.utils import *
import os

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load(os.path.join('ADVImage','data_mixup_all.npy'))
        labels = np.load(os.path.join('ADVImage','label_mixup_all.npy'))
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        #image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)
# addPepperNoise = AddPepperNoise(0.4)
# addGaussianNoise = AddGaussianNoise(mean=0.0,variance=20)
# dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)
dataset = MyDataset(None)
images = []
soft_labels = []
pepperNoise=[0.3,0.2,0.1]
# for _ in range(2):
for image, label in dataset:
    image = np.array(image)
    # # image= random_noise(image,noise_num=int(32*32*0.3))
    if random.random()<=0.5:
        image=gasuss_noise(image,var=0.008)
    # elif random.random()>0.6:
    #     image = sp_noise(image, 0.3)
    # elif 0.4<random.random()<=0.5:
    #     image = sp_noise(image, 0.2)
    else:
        # image = sp_noise(image, 0.1)
    # image=addPepperNoise(image)
        index=random.randint(0,2)
    #     image = sp_noise(image,0.35)
        image = sp_noise(image, pepperNoise[index])
    images.append(image)
    #soft_label = np.zeros(10)
     #+(0.1/9)
    # soft_label[label] += 1-0.1-(0.1/9)
    #soft_label[label] += random.uniform(0, 10) # an unnormalized soft label vector
    soft_labels.append(label)
dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)

for image, label in dataset:
    image = np.array(image)
    # # image= random_noise(image,noise_num=int(32*32*0.3))
    if random.random()<=0.5:
        image=gasuss_noise(image,var=0.008)
    else:
        index=random.randint(0,2)
        image = sp_noise(image, pepperNoise[index])
    images.append(image)
    soft_label = np.zeros(10)
     #+(0.1/9)
    # soft_label[label] += 1-0.1-(0.1/9)
    soft_label[label] += random.uniform(0, 10) # an unnormalized soft label vector
    soft_labels.append(soft_label)
images = np.array(images)
soft_labels = np.array(soft_labels)
print(images.shape, images.dtype, soft_labels.shape, soft_labels.dtype)

import os
current_path = 'PepperNoise'
if not os.path.exists(current_path):
    os.mkdir(current_path)
np.save(os.path.join(current_path,'data_3per_008gs_mixupAndTest_2w.npy'), images)
np.save(os.path.join(current_path,'label_3per_008gs_mixupAndTest_2w.npy'), soft_labels)