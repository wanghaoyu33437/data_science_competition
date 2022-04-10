import numpy as np
import torch
from tqdm import tqdm
import os

import segmentation_models_pytorch as seg
from segmentation_models_pytorch.losses import DiceLoss,SoftBCEWithLogitsLoss,SoftCrossEntropyLoss,LovaszLoss
# from losses.dice import DiceLoss
# from losses.lovasz import LovaszLoss
from torch.utils.data import DataLoader
from main import GIID_Model,GIID_Dataset
os.environ['CUDA_VISIBLE_DEVICES']='0'
import cv2
from PIL import Image
import  torch.backends.cudnn
# torch.backends.cudnn.benchmark=True
from utils import metric,SoftDiceLoss,FocalLoss
from torchvision import transforms
from transform import TTA



transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            # transforms.Normalize((0.87185234, 0.86747855, 0.8583319), (0.15480465, 0.16086526, 0.16299605)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

# test_path = '../../baseline/data/test/img/'
# train_path = '../../baseline/data/train/img/'
# mask_path =  '../../baseline/data/train/mask/'
# train_paths = sorted(os.listdir(train_path))
# mask_paths =sorted(os.listdir(mask_path))
# backbone_arch = 'seresnext50'#senet154 , seresnext50,Unet_eff_b4,senet154,seresnext50,efficientnet-b4，Upp_se_resnet152，LinkNet_se_resnet152
# model.cuda()
model = GIID_Model(backbone_arch='senet154',pretrained=False).cuda()
# model.train()
print('loading')
model_path = 'Mar_21_AllImg_s3data_1k_bookscreen_1446_1k_size768_newDataAug_senet154_bce1_dice02/tmp_epoch_093_train1.7385_val1.0900_valLoss0.2158/'
# model_path = 'Mar_7_AllImg_decompose768_adamw_CAWR_bec09_dice03_pathsize768/best_model_1.0882/'
# model_path = 'Mar_3_AllImg_S3_500_adamw_BCE07_Dice03/best_model_0.9263/'
model.load(model_path)
print('loading successful')
model.eval()
tta = TTA()
print(model_path)
#
size = 896
print("Size:%d"%size)
# train_npy = 'train_768.npy'
val_npy = 'val_%d.npy'%size
# train_file = np.load('../user_data/flist1/' + train_npy)
val_file = np.load('../user_data/flist1/' + val_npy)[:850]
train_npy = 'train_s3data_1800.npy'
# val_npy = 'val_400.npy'
train_file = np.load('../user_data/flist_add/' + train_npy)
train_file = np.concatenate([train_file, np.load('../user_data/flist_add/' + 'val_s3data_200.npy')])
# # val_file = np.load('../user_data/flist_noDeco/' + val_npy)
# train_file = np.concatenate([train_file, np.load('../user_data/flist1/' + 'train_s3_768.npy')])
# val_file = np.concatenate([val_file, np.load('../user_data/flist1/' + 'val_s3_768.npy')])
# train_file = np.concatenate([train_file, np.load('../user_data/flist_add/' + 'train_erasing_90.npy')])
val_file = np.concatenate([val_file, np.load('../user_data/flist_add/' + 'val_erasing_10.npy')])
train_num = len(train_file)
# val_num = len(val_file)
test_dataset = GIID_Dataset(train_num, train_file, choice='val')
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=4)

''' test loss'''
# d1 = SoftDiceLoss()
# bce = torch.nn.BCELoss()
# # d2 = DiceLoss('binary',from_logits=False)
# # L = LovaszLoss('binary',from_logits=False)
# focal =FocalLoss(mode='binary',alpha=1,gamma=2)
# save_path = '../save_out/test/'+backbone_arch+'/'+model_path.split('/')[0][-6:]+'_'+model_path.split('/')[1][:]

# with torch.no_grad():
#     scores = 0
#     f1, iou, gen_losses = [], [], []
#     for img, mask,_ in tqdm(test_loader):  # train
#         img = img.cuda()
#         mask = mask.cuda()
#         Mo = model(img)
#         # break
#         # Mo = Mo.sigmoid()
#         Mo = Mo * 255.
#         Mo = Mo.permute(0, 2, 3, 1).cpu().detach().numpy()
#         Mo[Mo < 100] = 0
#         Mo[Mo >= 100] = 255
#         # Mo = cv2.resize(Mo, (ori_size[1], ori_size[0]))
#         # save_seg_path = os.path.join(save_path, 'images', os.path.split(img_name)[-1].split('.')[0] + '.png')
#         # os.makedirs(os.path.split(save_seg_path)[0], exist_ok=True)
#         mask = mask * 255.
#         mask = mask.permute(0, 2, 3, 1).cpu().detach().numpy()
#         for i in range(len(img)):
#             a, b = metric(Mo[i] / 255, mask[i] / 255)
#             f1.append(a)
#             iou.append(b)
#
#             scores += (a + b)
# print(scores)

def output(model,input,ori_size,size=512):
    img = cv2.resize(input, (size, size))
    # img = input.resize((size,size))
    input = transform(img).unsqueeze(0).cuda().half()
    Mo = model(input)
    Mo = Mo * 255.
    Mo = Mo.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
    # Mo = np.array(Image.fromarray(Mo.astype(np.uint8)[:, :, 0]).resize((ori_size[0], ori_size[1]),Image.BILINEAR))
    Mo = cv2.resize(Mo.astype(np.float32), (ori_size[1], ori_size[0]))
    return Mo

well_example=[]
with torch.no_grad():
    scores = 0
    s1, s2, s3,s4 = [], [], [],[]
    f1, iou, gen_losses = [], [], []
    for img_name, mask_name in tqdm(val_file[:]):  # train
        img = cv2.imread(img_name)[:, :, ::-1]
        # img =Image.open(img_name)
        mask = np.array(Image.open(mask_name))
        if len(mask.shape)==2:
            mask = mask
        else:
            mask= mask[:,:,0]
        ori_size = img.shape
        # ori_size = img.size  # (W,H)
        #
        img = img.astype('float') / 255
        masks=[]
        Mo = output(model, img, ori_size, size)
        Mo[Mo < 100] = 0
        Mo[Mo >= 100] = 255
        masks.append(Mo.astype(np.uint8))
        f1, iou = metric(Mo / 255, mask / 255)
        if (f1+iou)>=1.3:
            well_example.append([img_name,mask_name])
        s1.append(f1+iou)

        # img1 = tta.HorizontalFlip(img)
        # Mo = output(model, img1, ori_size,size)
        # Mo[Mo < 100] = 0
        # Mo[Mo >= 100] = 255
        # Mo = tta.HorizontalFlip(Mo)
        # masks.append(Mo.astype(np.uint8))
        # f1, iou = metric(Mo / 255, mask / 255)
        # s2.append(f1 + iou)
        # #
        # mask_and = masks[0]
        # mask_or = masks[0]
        # #
        # for i in range(1, len(masks)):
        #     mask_and = mask_and & masks[i]
        # for i in range(1, len(masks)):
        #     mask_or = mask_or | masks[i]
        # f1, iou = metric(mask_and / 255, mask / 255)
        # s3.append(f1 + iou)
        # f1, iou = metric(mask_or / 255, mask / 255)
        # s4.append(f1+iou)

print(np.sum(s1),np.sum(s2),np.sum(s3),np.sum(s4))
# np.save('../user_data/val_And_s3_768_well%d_1.3.npy'%len(well_example),well_example)

