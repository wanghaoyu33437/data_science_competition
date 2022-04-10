import os
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
from models.unet import SCSEUnet
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class GIID_Model(nn.Module):
    def __init__(self,backbone_arch='seresnext50',date='',params='',pretrained=True):
        super(GIID_Model, self).__init__()
        self.lr = 3e-4
        self.backbone_arch = backbone_arch
        print(backbone_arch)
        self.networks = SCSEUnet(backbone_arch=backbone_arch,pretrained=pretrained) # seresnext50 , senet154
        self.gen = nn.DataParallel(self.networks).cuda().half()
        self.gen_optimizer = optim.AdamW(self.gen.parameters(), lr=3e-4,weight_decay=5e-4, betas=(0.9, 0.999))
        self.gen_scheduler = CosineAnnealingWarmRestarts(self.gen_optimizer , T_0=3, T_mult=2, eta_min=1e-6, last_epoch=-1)
    def forward(self, Ii):
        return self.gen(Ii)
    def backward(self, gen_loss=None):
        if gen_loss:
            gen_loss.backward(retain_graph=False)
            self.gen_optimizer.step()
            self.gen_optimizer.zero_grad()
    def lr_scheduler(self):
        self.gen_scheduler.step()
    def save(self, path=''):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.gen.state_dict(), self.save_dir + path + 'GIID_weights.pth')
    def load(self, path=''):
        self.gen.load_state_dict(torch.load(self.save_dir + path + 'GIID_weights.pth'))
def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path,exist_ok=True)

def decompose(data_path='../tianchi_data/data/test/',path_out = '../user_data/decompose/test_decompose/'):
    path = data_path
    flist = sorted(os.listdir(path))
    size_list = [896]
    rm_and_make_dir(path_out)
    for file in tqdm(flist):
        img = cv2.imread(path + file)
        H, W, _ = img.shape
        size_idx= 0
        while size_idx <len(size_list)-1:
            if H<size_list[size_idx+1] or W < size_list[size_idx+1]:
                break
            size_idx += 1
        size = size_list[size_idx]
        X, Y = H // size + 1, W // size + 1
        idx = 0
        for x in range(X-1):
            for y in range(Y-1):
                img_tmp = img[x * size: (x + 1) * size, y * size: (y + 1) * size, :]
                cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
                idx += 1
            img_tmp = img[x * size: (x + 1) * size, -size:, :]
            cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
            idx += 1
        for y in range(Y - 1):
            img_tmp = img[-size:, y * size: (y + 1) * size, :]
            cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
            idx += 1
        img_tmp = img[-size:, -size:, :]
        cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
        idx += 1
    return path_out

def merge(path,size_list):
    test_path = '../tianchi_data/data/test/'
    # rm_and_make_dir(path)
    val_lists =os.listdir(test_path)
    for file in tqdm(val_lists):
        img = cv2.imread(test_path + file)
        H, W, _ = img.shape
        size_idx =0
        while size_idx <len(size_list)-1:
            if H<size_list[size_idx+1] or W < size_list[size_idx+1]:
                break
            size_idx += 1
        size = size_list[size_idx]
        X, Y = H // size + 1, W // size + 1
        idx = 0
        rtn = np.zeros((H, W, 3), dtype=np.uint8)
        for x in range(X-1):
            for y in range(Y-1):
                img_tmp = cv2.imread(path + file[:-4] + '_%03d.png' % idx)
                os.remove(path + file[:-4] + '_%03d.png' % idx)
                rtn[x * size: (x + 1) * size, y * size: (y + 1) * size, :] += img_tmp
                idx += 1
            img_tmp = cv2.imread(path + file[:-4] + '_%03d.png' % idx)
            os.remove(path + file[:-4] + '_%03d.png' % idx)
            rtn[x * size: (x + 1) * size, -size:, :] += img_tmp
            idx += 1
        for y in range(Y - 1):
            img_tmp = cv2.imread(path + file[:-4] + '_%03d.png' % idx)
            os.remove(path + file[:-4] + '_%03d.png' % idx)
            rtn[-size:, y * size: (y + 1) * size, :] += img_tmp
            idx += 1
        img_tmp = cv2.imread(path + file[:-4] + '_%03d.png' % idx)
        os.remove(path + file[:-4] + '_%03d.png' % idx)
        rtn[-size:, -size:, :] += img_tmp
        idx += 1
        rtn[rtn>=200]=255
        cv2.imwrite(path + file[:-4] + '.png', rtn)
