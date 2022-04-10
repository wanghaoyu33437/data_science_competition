import numpy as np
import torch
import glob
from tqdm import tqdm
import os
import segmentation_models_pytorch as seg
from torch.utils.data import DataLoader
from main import GIID_Model,GIID_Dataset
os.environ['CUDA_VISIBLE_DEVICES']='2'
import cv2
from PIL import Image

from utils import metric
backbone_arch = 'senet154'#senet154 , seresnext50
model = GIID_Model(backbone_arch=backbone_arch).cuda()
model_path = 'Mar_3_AllImg_S3_500_adamw_BCE07_Dice03/best_model_0.9263/'
model.load(model_path)
model.eval()
from torchvision import transforms

transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            # transforms.Normalize((0.87185234, 0.86747855, 0.8583319), (0.15480465, 0.16086526, 0.16299605)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
size = 896
img_path = '../data/train_512_640_768_896_1024/img_%d/'%size
mask_path = '../data/train_512_640_768_896_1024/mask_%d/'%size

img_paths = sorted(os.listdir(img_path))[:500]
mask_paths =sorted(os.listdir(mask_path))[:500]
# name = []
# for a in img_paths:
#     if a[:-8] not in name :
#         name.append(a[:-8])
# # len(name)
# # train768='/home2/WHY/04_FakeImageDetection/Tianchi_FFT2_rank3/data/train_768/'
# # mask768='/home2/WHY/04_FakeImageDetection/Tianchi_FFT2_rank3/data/mask_768/'
# #
# # img_768_path = []
# # for n in name:
# #     l = glob.glob(train768 + n + '_*')
# #     img_768_path.extend(l)


# backbone_arch = 'seresnext50'#senet154 , seresnext50
# model.cuda()
# save_path = '../save_out/test/'+backbone_arch+'/'+model_path.split('/')[0][-6:]+'_'+model_path.split('/')[1][:]

# scores = 0
# f1, iou, gen_losses = [], [], []
# H,W=[],[]
# with torch.no_grad():
#     for name in tqdm(img_paths):  # test
#         # img = Image.open(test_path+img_name)
#         img = cv2.imread(img_path + name)[:, :, ::-1]
#         mask_gt = np.array(Image.open(mask_path+name))[:,:,0]
#         ori_size = img.shape
#         H.append(ori_size[1])
#         W.append(ori_size[0])
#         # print(ori_size)
#         img = img.astype('float') / 255
#         img = cv2.resize(img, (size,size))
#         inputs = transform(img).unsqueeze(0).cuda()
#         Mo = model(inputs)
#         Mo = Mo * 255.
#         Mo = Mo.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
#         Mo[Mo < 100] = 0
#         Mo[Mo >= 100] = 255
#         Mo = cv2.resize(Mo, (ori_size[1], ori_size[0]))
#         # os.makedirs(os.path.split(save_seg_path)[0], exist_ok=True)
#         f1, iou = metric(Mo/255,mask_gt/255)
#         scores+=(f1+iou)
# print(scores/len(img_paths))
# # print('mean Height:',np.mean(H),'Mean Width:',np.mean(W))
scores = 0
train_path = '../data/train_896/img_896/'
train_paths = sorted(os.listdir(train_path))[:600]
mask_path = '../data/train_896/mask_896/'
mask_paths = sorted(os.listdir(mask_path))
with torch.no_grad():
    for name in tqdm(train_paths):  # test
        # img = Image.open(test_path+img_name)
        img = cv2.imread(train_path + name)[:, :, ::-1]
        mask_gt = np.array(Image.open(mask_path+name))[:,:,0]
        ori_size = img.shape
        # print(ori_size)
        img = img.astype('float') / 255
        img = cv2.resize(img, (896,896))
        inputs = transform(img).unsqueeze(0).cuda()
        Mo = model(inputs)
        Mo = Mo * 255.
        Mo = Mo.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
        Mo[Mo < 100] = 0
        Mo[Mo >= 100] = 255
        Mo = cv2.resize(Mo, (ori_size[1], ori_size[0]))
        # os.makedirs(os.path.split(save_seg_path)[0], exist_ok=True)
        f1, iou = metric(Mo/255,mask_gt/255)
        scores+=(f1+iou)
print(scores/len(train_paths))
