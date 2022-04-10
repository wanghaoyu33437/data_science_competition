import os
import cv2
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.unet import SCSEUnet
from utils import metric,rm_and_make_dir,SoftDiceLoss,AverageMeter,FocalLoss,decompose
from data_augmentation import rand_bbox,copy_move,random_paint
from transform import gen_transform
from config import params_config,model_configs
import torch.backends.cudnn

import torch.cuda
import logging
gpu_ids = '0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
# torch.cuda.set_per_process_memory_fraction(0.98, 0)
# torch.cuda.set_per_process_memory_fraction(0.98, 1)
# torch.cuda.set_per_process_memory_fraction(0.88, 2)
# torch.cuda.set_per_process_memory_fraction(0.88, 3)

seed = 7777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class GIID_Dataset(Dataset):
    def __init__(self, num=0, file=None, choice='train', test_path='',model_args=None):
        self.num = num
        self.choice = choice
        if self.choice == 'test':
            self.test_path = test_path
            self.filelist = sorted(os.listdir(self.test_path))
        else:
            self.filelist = file
        self.train_transform= gen_transform(resize=model_args['train_size'],mode='train')
        self.val_transform = gen_transform(resize=model_args['train_size'],mode='val')
        self.copy_move = model_args['transform']['Copy_move']
        self.paint = model_args['transform']['random_paint']
    def __getitem__(self, idx):
        return self.load_item(idx)
    def __len__(self):
        if self.choice == 'test':
            return len(self.filelist)
        return self.num

    def load_item(self, idx):
        fname1, fname2 = self.filelist[idx]
        img = cv2.imread(fname1)[..., ::-1]
        H, W, _ = img.shape
        if fname2 == '':
            mask = np.zeros([H, W, 3])
        else:
            mask = cv2.imread(fname2)
        H, W, _ = img.shape
        if self.choice == 'train':
            ''' '''
            i = np.random.rand(1)
            if i <= self.paint:
                img,mask = random_paint(img,mask)
            transformed = self.train_transform(image=img,mask=mask)
            img,mask= transformed['image'],transformed['mask']
            mask = mask[:,:,:1].permute(2, 0, 1).float()/255
            i = np.random.rand(1)
            if i <= self.copy_move:
                if random.random()<=0.5: # 从其他图像copy一块
                    random_index = np.random.randint(self.num)
                    while random_index==idx:
                        random_index = np.random.randint(self.num)
                    fname1, fname2 = self.filelist[random_index]
                    cut_img = cv2.imread(fname1)[..., ::-1]
                    transformed_cutmix = self.val_transform(image=cut_img)
                    cut_img = transformed_cutmix['image']
                    img,mask = copy_move(img,cut_img,mask)
                else:
                    img, mask = copy_move(img, None, mask)
        elif self.choice == 'val':
            transformed = self.val_transform(image= img,mask=mask)
            img,mask= transformed['image'],transformed['mask']
            mask = mask[:, :, :1].permute(2, 0, 1).float() / 255

        return img , mask, fname1.split('/')[-1]

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)


class GIID_Model(nn.Module):
    def __init__(self,backbone_arch='senet154',pretrained=True,model_args = None):
        super(GIID_Model, self).__init__()
        self.lr = params_config['optimizer_hyperparameters']['lr']
        self.backbone_arch = backbone_arch
        self.networks = SCSEUnet(backbone_arch=backbone_arch,pretrained=pretrained) # seresnext50 , senet154
        self.gen = nn.DataParallel(self.networks).cuda()
        self.gen_optimizer = optim.__dict__[params_config['optimizer_name']](self.gen.parameters(),
                                               **params_config['optimizer_hyperparameters'])
        self.gen_scheduler=torch.optim.lr_scheduler.__dict__[params_config['scheduler_name']](self.gen_optimizer,
                                                                  **params_config['scheduler_hyperparameters'])
        self.save_dir = '../user_data/model_data/'+model_args['name']
        self.bce_loss = nn.BCELoss()
        self.dice_loss = SoftDiceLoss()
        self.focal_loss = FocalLoss(mode='binary', alpha=0.25, gamma=2)
    def process(self, Ii, Mg):
        Mo = self(Ii)
        bce_loss = self.bce_loss(Mo.view(Mo.size(0), -1), Mg.view(Mg.size(0), -1))
        focal_loss  = self.focal_loss(Mo,Mg)
        dice_loss = self.dice_loss(Mo,Mg)
        # 返回 F，D，B 格式
        return Mo, focal_loss,dice_loss,bce_loss
    def forward(self, Ii):
        return self.gen(Ii)
    def backward(self, gen_loss=None):
        if gen_loss:
            gen_loss.backward(retain_graph=False)
            self.gen_optimizer.step()
            self.gen_optimizer.zero_grad()
    def lr_scheduler(self):
        self.gen_scheduler.step()
    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.gen.state_dict(), self.save_dir + '/GIID_weights.pth')
    def load(self):
        self.gen.load_state_dict(torch.load(self.save_dir + '/GIID_weights.pth'))

class ForgeryForensics():
    def __init__(self, backbone_arch='senet154',model_args=None):
        self.test_num = 1
        self.batch_size = model_args['batch_size']
        self.train_npy = model_args['data_path']
        self.train_file = np.load(self.train_npy)
        self.train_num = len(self.train_file)
        print("train_num:%d"%self.train_num)
        train_dataset = GIID_Dataset(self.train_num, self.train_file, choice='train',model_args=model_args)
        print("loading model")
        self.giid_model = GIID_Model(backbone_arch=backbone_arch,model_args=model_args).cuda()
        print("loading successful")
        self.n_epochs = model_args['epoch']
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,pin_memory=True)
    def train(self):
        cnt, gen_losses, f1, iou = 0, [], [], []
        for epoch in range(1, self.n_epochs):
            f_avg =AverageMeter()
            d_avg = AverageMeter()
            b_avg = AverageMeter()
            for items in tqdm(self.train_loader):
                cnt += self.batch_size
                self.giid_model.train()
                Ii, Mg = (item.cuda() for item in items[:-1])
                Mo, f_loss,dice_loss,bce_loss = self.giid_model.process(Ii, Mg)
                loss = 0.7*bce_loss+0.3*dice_loss+1*f_loss
                self.giid_model.backward(loss)
                gen_losses.append(loss.item())
                f_avg.update(f_loss.item() if f_loss!=0 else 0,Mo.size(0))
                d_avg.update(dice_loss.item()if dice_loss!=0 else 0,Mo.size(0))
                b_avg.update(bce_loss.item()if bce_loss!=0 else 0,Mo.size(0))
                Mg, Mo = self.convert2(Mg), self.convert2(Mo)
                Mo[Mo < 127.5] = 0
                Mo[Mo >= 127.5] = 255
                for i in range(len(Ii)):
                    a, b = metric(Mo[i] / 255, Mg[i] / 255)
                    f1.append(a)
                    iou.append(b)
                if cnt % (self.batch_size*10)==0:
                    logging.info('epoch:%d,focal_loss:%5.4f,dice_loss:%5.4f,bce_loss:%5.4f' % (epoch,f_avg.avg,d_avg.avg,b_avg.avg)) #np.mean(gen_losses,axis=0)[1]
                if cnt % (self.batch_size*50)==0:
                    logging.info('Tra (%d/%d): Loss:%5.4f F1:%5.4f IOU:%5.4f SUM:%5.4f'
                      % (cnt, self.train_num, np.mean(gen_losses), np.mean(f1), np.mean(iou), np.mean(f1) + np.mean(iou)))
            self.giid_model.lr_scheduler()
            cnt, gen_losses, f1, iou = 0, [], [], []
        self.giid_model.save()
    def convert1(self, img):
        img = img * 127.5 + 127.5
        img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
        return img
    def convert2(self, x):
        x = x * 255.
        return x.permute(0, 2, 3, 1).cpu().detach().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, default=0, help='which model train')
    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    print('Decomposing trainset')
    decompose(data_path='../tianchi_data/data/train/',path_out = '../user_data/decompose/train_768/')
    decompose(data_path='../tianchi_data/data/train_mask/', path_out='../user_data/decompose/mask_768/')
    print('Decomposing trainset finish')
    model = ForgeryForensics(backbone_arch='senet154',model_args=model_configs[args.model]) # senet154,seresnext50
    os.makedirs(model.giid_model.save_dir,exist_ok=True)
    logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.DEBUG,
            handlers=[
                logging.FileHandler(os.path.join(model.giid_model.save_dir, 'train.log')),
                logging.StreamHandler()
        ])
    logging.info(model_configs[args.model])  # senet154
    model.train()