from __future__ import print_function
import torch
from PIL import Image
# import matplotlib.pyplot as plt
import foolbox as fb
from foolbox.criteria import TargetedMisclassification
from utils import load_model
import time
import os
import numpy as np
import torchvision.transforms as transforms
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = 'cuda'

v = np.array([0.2023, 0.1994, 0.2010])
m = np.array([0.4914, 0.4822, 0.4465])

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = load_model('resnet50')
    model.load_state_dict(checkpoint['state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load('data_test.npy')
        labels = np.load('label_test.npy')
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
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
import torchvision
trainset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform_train)

images = []
soft_labels = []
one_hots = []
for image, label in trainset:
    image = np.array(image)
    images.append(image)
    soft_labels.append(label)
    one_hot = np.zeros(10)
    one_hot[label] += random.uniform(0, 10) # an unnormalized soft label vector
    one_hots.append(one_hot)

model = load_checkpoint('AllImage_adv7k_pepper035_Gaussian008_93_49/resnet50.pth.tar')
model.to(device)
# 通过fb.PyTorchModel封装成的类，其fmodel使用与我们训练的simple_model基本一样
fmodel = fb.PyTorchModel(model,bounds=(0,1))
# fmodel = fb.PyTorchModel(model)
# 如下代码dataset可以选择cifar10,cifar100,imagenet,mnist等。其图像格式是channel_first
# 由于fmodel设置了bounds，如下代码fb.utils.samples获得的数据，会自动将其转化为bounds范围中
image_save = []
for i in range(0,10000,10):
    images_i = torch.tensor(images[i:i+10]).cuda()
    labels = torch.tensor(soft_labels[i:i+10]).int().cuda()
    # print (images_i.shape)
    # print (labels.shape)

    #labels_orinal = np.load('label.npy')
    acc = fb.utils.accuracy(fmodel, images_i, labels)
    # print(acc)

    # image = images.cpu().numpy().transpose((0, 2, 3, 1))
    # image = image * v + m

    # plt.figure('test')
    # for i in range(1,11):
    #     plt.subplot(2,5,i)
    #     plt.imshow(image[i-1])

    attack = fb.attacks.VirtualAdversarialAttack(steps=5)
    raw, clipped, is_adv = attack(fmodel, images_i, labels, epsilons=0.8)
    image_adv = clipped.cpu().numpy().transpose((0, 2, 3, 1))

    image_adv = image_adv * v + m
    image_adv = (image_adv*255).astype(np.uint8)
    # plt.figure('test1')
    # for i in range(1,11):
    #     plt.subplot(2,5,i)
    #     plt.imshow(image_adv[i-1])

    # plt.figure('test2')
    # image_diff = image_adv-image
    # for i in range(1,11):
    #     plt.subplot(2,5,i)
    #     plt.imshow(image_diff[i-1])

    # plt.show()
    #
    image_save.extend(image_adv)
    #print(images.shape, images.dtype)
    print(i)
current_path = 'ADVImage'

if not os.path.exists(current_path):
    os.mkdir(current_path)

image_save = np.array(image_save)
one_hots = np.array(one_hots)
print(image_save.shape, image_save.dtype, one_hots.shape, one_hots.dtype)
np.save(os.path.join(current_path,'data.npy'), images)
np.save(os.path.join(current_path,'label.npy'), one_hots)

