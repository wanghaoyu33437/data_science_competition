from math import exp
import numpy as np
from torchvision import transforms,models
import os
import torch
from torch import nn
from PIL import Image
import cv2

infer_transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

IMAGES_KEY = 'images'
MODEL_INPUT_KEY = 'images'
LABEL_OUTPUT_KEY = 'predicted_label'
MODEL_OUTPUT_KEY = 'scores'
LABELS_FILE_NAME = 'labels.txt'


def decode_image(file_content):
    image = Image.open(file_content)
    image = image.convert('RGB')
    return image


def read_label_list(path):
    with open(path, 'r', encoding="utf8") as f:
        label_list = f.read().split(os.linesep)
    label_list = [x.strip() for x in label_list if x.strip()]
    return label_list


def resnet50(model_path):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    # model.load_state_dict(torch.load(model_path))

    model.eval()

    return model


def predict(file_name):
    LABEL_LIST = read_label_list('./model/labels.txt')
    model = resnet50('./model/model.pth')

    image1 = decode_image(file_name)

    input_img = infer_transformation(image1)

    input_img = torch.autograd.Variable(torch.unsqueeze(input_img, dim=0).float(), requires_grad=False)

    logits_list = model(input_img)[0].detach().numpy().tolist()
    print(logits_list)
    maxlist = max(logits_list)
    print(maxlist)

    z_exp = [exp(i - maxlist) for i in logits_list]

    sum_z_exp = sum(z_exp)
    softmax = [round(i / sum_z_exp, 3) for i in z_exp]
    print(softmax)
    labels_to_logits = {
        LABEL_LIST[i]: s for i, s in enumerate(softmax)
    }

    predict_result = {
        LABEL_OUTPUT_KEY: max(labels_to_logits, key=labels_to_logits.get),
        MODEL_OUTPUT_KEY: labels_to_logits
    }

    return predict_result


file_name = './data/test/ASC-US&LSIL/03424.jpg'
result = predict(file_name)  # 可以替换其他图片
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))  # 设置窗口大小
img = decode_image(file_name)
plt.imshow(img)
plt.show()

print(result)