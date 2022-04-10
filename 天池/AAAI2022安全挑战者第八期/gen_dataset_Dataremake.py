import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import numpy as np
import torchvision
import random

# from torchvision import transforms
from models.googlenet import *
from models.resnet import *
from models.vgg import *
from models.senet import SENet18
from models.mobilenet import  *
from models.mobilenetv2 import *
from models.senet import *
from models.densenet import densenet121
from PIL import Image
from Bag_trick.utils import *
import numpy as np
from tqdm import tqdm
from torchvision import  transforms
import torch
import torch.utils.data


# data = np.load('AllImage_adv7k_pepper035_Gaussian008_93_49/data.npy')
# label = np.load('AllImage_adv7k_pepper035_Gaussian008_93_49/label.npy')
# images = list(data)
# labels = list(label)

#
#
# data3 = np.load('PepperNoise/data_060new.npy')
# label3 = np.load('GaussianNoise/label.npy')
# images = list(data3)
# labels = list(label3)

data3 = np.load('PepperNoise/data_3per_008gs_mixupAndTest_2w.npy')
label3 = np.load('PepperNoise/label_3per_008gs_mixupAndTest_2w.npy')
images = list(data3)
labels = list(label3)

data2 = np.load('ADVImage/data_adv_3w_min003.npy')
label2 = np.load('ADVImage/label_adv_3w_min003.npy')
# images.extend(data2[:10000:])
# labels.extend(label2[:10000:])
# # images.extend(data2[10000:20000:])
# # labels.extend(label2[10000:20000:])
# images.extend(data2[20000::])
# labels.extend(label2[20000::])
images.extend(data2)
labels.extend(label2)
#
# data1 = np.load('ADVImage/data_mixup_all.npy')
# label1 = np.load('ADVImage/label_mixup_all.npy')
# images.extend(data1)
# labels.extend(label1)

#
# data1 = np.load('PepperNoise/data_040new.npy')
# # label1 = np.load('PepperNoise/label010.npy')
# images.extend(data1)
# labels.extend(label3)
# #
# data2 = np.load('PepperNoise/data_020new.npy')
# # label2 = np.load('PepperNoise/label020.npy')
# images.extend(data2)
# labels.extend(label3)




# data4 = np.load('GaussianNoise/data008.npy')
# # # label4 = np.load('GaussianNoise/label010.npy')
# images.extend(data4)
# labels.extend(label3)
#
# #
# data5 = np.load('ADVImage/data8_2.npy')
# label5 = np.load('ADVImage/label_sm01.npy')
# images.extend(data5)
# labels.extend(label5)

# data6 = np.load('RandomNoise/data_060new.npy')
# # label6 = np.load('RandomNoise/label030.npy')
# images.extend(data6)
# labels.extend(label6)
# num=[0]*10
# for d ,l  in zip(data5,label5):
#     index = l.argmax()
#     if num[index] >=700:
#         continue
#     num[index] += 1
#     images.append(d)
#     labels.append(l)






#
# #
images = np.array(images)
hard_labels  = np.array(labels)
print(images.shape, images.dtype, hard_labels.shape, hard_labels.dtype)
current_path = 'AllImage_FGSM003_015_030_3w_pepper060040020AndGaussian008_mixupAndTest_2w'
if not os.path.exists(current_path):
    os.mkdir(current_path)
np.save(os.path.join(current_path,'data.npy'), images)
np.save(os.path.join(current_path,'label.npy'), hard_labels)