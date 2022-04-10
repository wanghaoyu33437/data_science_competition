import numpy as np
import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from main import GIID_Model
os.environ['CUDA_VISIBLE_DEVICES']='2'
import cv2
from PIL import Image
from utils import metric
from transform import TTA
size = 896
train_npy = 'train_768.npy'
val_npy = 'val_%d.npy'%size
train_file = np.load('../user_data/flist1/' + train_npy)
val_file = np.load('../user_data/flist1/' + val_npy)[:850]
# test_paths = sorted(os.listdir(test_path))
# mask_paths =sorted(os.listdir(mask_path))
# test_dataset = GIID_Dataset(len(os.listdir(test_path)), choice='test',test_path=test_path)
# test_loader = DataLoader(test_dataset,batch_size=16,num_workers=4,pin_memory=True)
backbone_arch = 'senet154'#senet154 , seresnext50
model = GIID_Model(backbone_arch=backbone_arch).cuda()
# model_path = 'Mar_3_AllImg_S3_500_adamw_BCE07_Dice03/best_model_0.9263/'
model_path = 'Mar_9_AllImg_decompose768_val500_s3_data500_adamw_CAWR_bec07_dice03_focal1_pathsize768/tmp_epoch_037_1.2995/'
# model_path = 'Mar_4_AllImg_s3_1k_adamw_bec06_dice04_pathsize640/best_model_0.8827/'
model.load(model_path)
model.eval()
tta = TTA()
from torchvision import transforms
transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            # transforms.Normalize((0.87185234, 0.86747855, 0.8583319), (0.15480465, 0.16086526, 0.16299605)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
def output(model,input,ori_size,size=512):
    img = cv2.resize(input, (size, size))
    input = transform(img).unsqueeze(0).cuda()
    Mo = model(input)
    Mo = Mo * 255.
    Mo = Mo.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
    Mo = cv2.resize(Mo, (ori_size[1], ori_size[0]))
    return Mo

# save_path = '../save_out/test/'+backbone_arch+'/'+model_path.split('/')[1][-6:]+"_AllImg_decompose768_adamw_bec07_dice03_pathsize768"
save_path = './'
bad_examples=[]
if __name__ == '__main__':
    with torch.no_grad():
        s1,s2,s3,s4,s5 ,s6,s7,s8 = [],[],[],[],[],[],[],[]
        s = []
        f1, iou, gen_losses = [], [], []
        for img_name,mask_name in tqdm(val_file[:]):  # test
            img = cv2.imread(img_name)[:,:,::-1]
            mask_gt = np.array(Image.open(mask_name))[:,:,0]
            ori_size = img.shape
            img = img.astype('float')/255
            resize=[896]
            masks=[]
            # for size in resize:
                # no flip
            Mo = output(model, img, ori_size, resize[0])
            Mo[Mo < 100] = 0
            Mo[Mo >= 100] = 255
            masks.append(Mo.astype(np.uint8))
            f1, iou = metric(Mo / 255, mask_gt / 255)
            s1.append(f1+iou)
            #                              os.path.split(img_name)[-1].split('.')[0] + '_%d_score_%5.4f.png' % (
            #                              size, f1 + iou))
            # os.makedirs(os.path.split(save_seg_path)[0], exist_ok=True)
            # cv2.imwrite(save_seg_path, Mo.astype(np.uint8))

            # img1 = tta.VerticalFlip(img)
            # Mo = output(model, img1, ori_size, resize[0])
            # Mo = tta.VerticalFlip(Mo)
            # Mo[Mo < 100] = 0
            # Mo[Mo >= 100] = 255
            # masks.append(Mo.astype(np.uint8))
            # f1, iou = metric(Mo / 255, mask_gt / 255)
            # s2.append(f1 + iou)


            img2 = tta.HorizontalFlip(img)
            Mo = output(model, img2, ori_size, resize[0])
            Mo = tta.HorizontalFlip(Mo)
            Mo[Mo < 100] = 0
            Mo[Mo >= 100] = 255
            masks.append(Mo.astype(np.uint8))
            f1, iou = metric(Mo / 255, mask_gt / 255)
            s3.append(f1+iou)
            #
            # img4 = tta.Transpose(img)
            # ori_size1 = img4.shape
            # Mo=output(model,img4,ori_size1,size)
            # Mo = tta.Transpose(Mo)
            # Mo[Mo < 100] = 0
            # Mo[Mo >= 100] = 255
            # masks.append(Mo.astype(np.uint8))
            # f1, iou = metric(Mo / 255, mask_gt / 255)
            # s4.append(f1+iou)
                # save_seg_path = os.path.join(save_path, 'test_taa', os.path.split(img_name)[-1].split('.')[0] + '_%d_VF_score_%5.4f.png'%(size,f1+iou))
                # cv2.imwrite(save_seg_path, Mo.astype(np.uint8))
            mask_and = masks[0]
            mask_or = masks[0]
            for i in range(1,len(masks)):
                mask_and = mask_and&masks[i]
            f1, iou = metric(mask_and / 255, mask_gt / 255)
            s.append(f1+iou)
            for i in range(1, len(masks)):
                mask_or = mask_or | masks[i]
            f1, iou = metric(mask_or / 255, mask_gt / 255)
            s5.append(f1 + iou)
            # save_seg_path = os.path.join(save_path, 'test_taa',
            #                              os.path.split(img_name)[-1].split('.')[0] + '_%d_score_and_%5.4f.png' % (
            #                              size, f1 + iou))
            # cv2.imwrite(save_seg_path, mask_and.astype(np.uint8))
            print('s_and:%5.4f,s_or:%5.4f,s_noflip: %5.4f,s_VF: %5.4f,s_HF:%5.4f ,s_TR: %5.4f'%(np.mean(s),np.mean(s5),np.mean(s1),np.mean(s2),np.mean(s3),np.mean(s4)))

