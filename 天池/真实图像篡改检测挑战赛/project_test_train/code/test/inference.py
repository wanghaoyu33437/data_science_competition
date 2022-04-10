import numpy as np
import torch
from tqdm import tqdm
import os
# from main import GIID_Model
import argparse
import cv2
from PIL import Image
from models.unet import SCSEUnet
from torchvision import transforms
from utils import rm_and_make_dir,merge,decompose
from torch import nn
import shutil
transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


# os.environ['CUDA_VISIBLE_DEVICES']='0'

def load_weight(path,model):
    model.load_state_dict(torch.load(os.path.join(path,'GIID_weights.pth')))
    model.eval()

def output(model,input,ori_size,size=512):
    img = cv2.resize(input, (size, size))
    input = transform(img).unsqueeze(0).cuda()
    Mo = model(input)
    Mo = Mo * 255.
    Mo = Mo.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
    Mo = cv2.resize(Mo.astype(np.float32), (ori_size[1], ori_size[0]))
    return Mo
def vote(save_decompose_path,model_folders,test_img):
    masks=[]
    for name in model_folders:
        mask = cv2.imread(os.path.join(save_decompose_path, name, test_img))
        mask[mask >= 100] = 255
        mask[mask < 100] = 0
        masks.append(mask)
    mask = np.zeros_like(mask,dtype=np.uint8)
    for m in masks:
        mask = mask+m/255
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',type=str,default='0',help='Select GPU device')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

    backbone_arch = 'senet154'  # senet154 , seresnext50.LinkNet_se_resnet152
    model = nn.DataParallel(SCSEUnet(backbone_arch=backbone_arch, pretrained=False)).cuda()  # seresnext50 , senet154

    size = 896
    print('Decomposing')
    test_path=decompose(data_path='../tianchi_data/data/test/')
    test_paths = sorted(os.listdir(test_path))
    save_decompose_path = '../user_data/tmp_test/'
    rm_and_make_dir(save_decompose_path)
    #
    model_path = '../user_data/model_data/'
    model_folders = os.listdir(model_path)
    with torch.no_grad():
        for f_name in model_folders:
            print("Loading %s"%f_name)
            load_weight(model_path+f_name,model)
            save_path = save_decompose_path+f_name+'/'
            for img_name in tqdm(test_paths[:]):  # test
                img = cv2.imread(test_path+img_name)[:,:,::-1]
                ori_size = img.shape
                img = img.astype('float')/255
                masks = []
                Mo = output(model, img, ori_size,size)
                Mo[Mo < 100] = 0
                Mo[Mo >= 100] = 255
                save_single_path = os.path.join(save_path,os.path.split(img_name)[-1].split('.')[0] + '.png')
                os.makedirs(os.path.split(save_single_path)[0], exist_ok=True)
                cv2.imwrite(save_single_path, Mo.astype(np.uint8))
            merge(save_path,[size])
    testImage_names = sorted(os.listdir(save_path))
    output_dir = '../prediction_result/images/'
    rm_and_make_dir(output_dir)
    for name in tqdm(testImage_names):
        mask=vote(save_decompose_path, model_folders,name)
        mask_or = mask.copy()
        mask_or[mask_or >= 1] = 255
        mask_or[mask_or < 1] = 0
        mask[mask >= 3] = 255 # 投票>=2 选中
        mask[mask < 3] = 0
        # cv2.imwrite(output_dir + name, mask_or.astype(np.uint8))
        if (mask==255).sum()<2000:
            cv2.imwrite(output_dir+name, mask_or.astype(np.uint8))
        else:
            cv2.imwrite(output_dir+name, mask.astype(np.uint8))
    print('Inference Finish')




