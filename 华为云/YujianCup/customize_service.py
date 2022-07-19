# !/usr/bin/python
# -*- coding: UTF-8 -*-


from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import os
from math import exp
import numpy as np

from PIL import Image
import cv2
import torch.nn as nn
from torch.nn import functional as F
import torch
import logging
import torchvision.models as models
import torchvision.transforms as transforms
import timm

infer_transformation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(384),
    # transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

IMAGES_KEY = 'images'
MODEL_INPUT_KEY = 'images'
LABEL_OUTPUT_KEY = 'predicted_label'
MODEL_OUTPUT_KEY = 'scores'
LABELS_FILE_NAME = 'labels.txt'


def decode_image(file_content):
    """
    Decode bytes to a single image
    :param file_content: bytes
    :return: ndarray with rank=3
    """
    image = cv2.imread(file_content)[:,:,::-1]
    # print(image.shape)
    # image = np.asarray(image, dtype=np.float32)
    return image


#    image_content = r.files[file_content].read() # python 'List' class that holds byte
#    np_array = np.fromstring(image_content, np.uint8) # numpy array with dtype np.unit8
#    img_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR) # numpy array in shape [height, width, channels]


def read_label_list(path):
    """
    read label list from path
    :param path: a path
    :return: a list of label names like: ['label_a', 'label_b', ...]
    """
    with open(path, 'r', encoding="utf8") as f:
        label_list = f.read().split(os.linesep)
    label_list = [x.strip() for x in label_list if x.strip()]
    # print(' label_list',label_list)
    return label_list


class KingmedCCPredictService(PTServingBaseService):
    def __init__(self, model_name, model_path):

        global LABEL_LIST
        super(KingmedCCPredictService, self).__init__(model_name, model_path)
        self.model = Convnext_b(model_path)
        dir_path = os.path.dirname(os.path.realpath(self.model_path))

        LABEL_LIST = read_label_list(os.path.join(dir_path, LABELS_FILE_NAME))

    def _preprocess(self, data):

        """
        `data` is provided by Upredict service according to the input data. Which is like:
          {
              'images': {
                'image_a.jpg': b'xxx'
              }
          }
        For now, predict a single image at a time.
        """
        preprocessed_data = {}
        input_batch = []
        for file_name, file_content in data[IMAGES_KEY].items():

            #           print('\tAppending image: %s' % file_name)

            image1 = decode_image(file_content)
            H,W,_ = image1.shape
            img1,img2 = image1[:,:int(W/2+0.5),:],image1[:,int(W/2+0.5):,:]

            if torch.cuda.is_available():
                input_batch.append(infer_transformation(img1).cuda())
                input_batch.append(infer_transformation(img2).cuda())
            else:
                input_batch.append(infer_transformation(img1))
                input_batch.append(infer_transformation(img2))
        input_batch_var = torch.autograd.Variable(torch.stack(input_batch, dim=0), volatile=True)
        preprocessed_data[MODEL_INPUT_KEY] = input_batch_var

        # print('preprocessed_data',input_batch_var.shape())

        return preprocessed_data

    def _postprocess(self, data):

        """
        `data` is the result of your model. Which is like:
          {
            'logits': [[0.1, -0.12, 0.72, ...]]
          }
        value of logits is a single list of list because one image is predicted at a time for now.
        """

        # logits_list = [0.1, -0.12, 0.72, ...]
        # data1,data2 = data['images'][0],data['images'][1]
        data = torch.sum(data['images'],dim=0)
        if torch.cuda.is_available():
            data = data.cpu()
        logits_list = data.detach().numpy().tolist()
        maxlist = max(logits_list)
        z_exp = [exp(i - maxlist) for i in logits_list]

        sum_z_exp = sum(z_exp)
        softmax = [round(i / sum_z_exp, 3) for i in z_exp]

        # labels_to_logits = {

        #     'label_a': 0.1, 'label_b': -0.12, 'label_c': 0.72, ...

        # }

        labels_to_logits = {
            LABEL_LIST[i]: s for i, s in enumerate(softmax)
            # LABEL_LIST[i]: s for i, s in enumerate(logits_list)
        }

        predict_result = {
            LABEL_OUTPUT_KEY: max(labels_to_logits, key=labels_to_logits.get),
            MODEL_OUTPUT_KEY: labels_to_logits
        }

        return predict_result

class F_interpolate(nn.Module):
    def __init__(self,size):
        self.size = size
        super(F_interpolate, self).__init__()
    def forward(self,inputs):
        return F.interpolate(inputs,(self.size,self.size))

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """
    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means)
        self.sds = torch.tensor(sds)

    def forward(self, input):
        (batch_size, num_channels, height, width) = input.shape
        # print(input.shape)
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
def Convnext_b(model_path, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = timm.create_model(model_name='swin_s3_tiny_224',**{'pretrained': False,'num_classes': 4})
    if torch.cuda.is_available():
        print('load model to gpu')
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    else:
        print('load model to cpu')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    if torch.cuda.is_available():
        model = model.to('cuda')

    return model
