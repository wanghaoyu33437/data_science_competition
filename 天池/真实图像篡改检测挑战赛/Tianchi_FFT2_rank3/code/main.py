import os
import cv2
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm
import glob
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from  torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler,autocast
from torchvision import transforms
from models.unet import SCSEUnet
from utils import metric,rm_and_make_dir,forensics_test_merge,decompose,\
    TverskyLoss,SoftDiceLoss,AverageMeter,FocalLoss
from data_augmentation import rand_bbox,copy_move,random_paint
from losses.lovasz import LovaszLoss,_lovasz_softmax
import segmentation_models_pytorch as seg
from transform import gen_transform
from segmentation_models_pytorch.losses import DiceLoss,SoftBCEWithLogitsLoss
from PIL import Image
import torch.backends.cudnn
# torch.backends.cudnn.benchmark = True
import torch.cuda
import logging
gpu_ids = '0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
# torch.cuda.set_per_process_memory_fraction(0.98, 0)
# torch.cuda.set_per_process_memory_fraction(0.98, 1)
# torch.cuda.set_per_process_memory_fraction(0.88, 2)
# torch.cuda.set_per_process_memory_fraction(0.88, 3)

# transform = transforms.Compose([
#             np.float32,
#             transforms.ToTensor(),
#             # transforms.Normalize((0.87185234, 0.86747855, 0.8583319), (0.15480465, 0.16086526, 0.16299605)),
#             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ])
seed = 7777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
patch_size = '768'  # The input size of training data


train_fold_best_val_record = np.load('train_fold_best_val_record.npy', allow_pickle=True)

class GIID_Dataset(Dataset):
    def __init__(self, num=0, file='', choice='train', test_path='', tta_idx=1):
        self.num = num
        self.choice = choice
        if self.choice == 'test':
            self.test_path = test_path
            self.filelist = sorted(os.listdir(self.test_path))
        else:
            self.filelist = file
        self.train_transform= gen_transform(resize=int(patch_size),mode='train')
        self.val_transform = gen_transform(resize=int(patch_size),mode='val')
        self.tta_idx = tta_idx
        self.copy_move = 0.0
        self.paint = 0.0
        self.test_img_path = sorted(glob.glob('../../baseline/data/test/img/*'))
        self.test_mask_path = sorted(glob.glob('../../baseline/data/test/best_2774/*'))
    def __getitem__(self, idx):
        return self.load_item(idx)

    def __len__(self):
        if self.choice == 'test':
            return len(self.filelist)
        return self.num

    def load_item(self, idx):
        if self.choice != 'test':
            fname1, fname2 = self.filelist[idx]
        else:
            fname1, fname2 = self.test_path + self.filelist[idx], ''
        # img = np.array(Image.open(fname1))
        img = cv2.imread(fname1)[..., ::-1]
        H, W, _ = img.shape
        if fname2 == '':
            mask = np.zeros([H, W, 3])
        else:
            mask = cv2.imread(fname2)
        if self.tta_idx == 5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        if self.tta_idx == 2 or self.tta_idx == 6:
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.cv2.ROTATE_90_CLOCKWISE)
        elif self.tta_idx == 3 or self.tta_idx == 7:
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask = cv2.rotate(mask, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.tta_idx == 4 or self.tta_idx == 8:
            img = cv2.rotate(img, cv2.cv2.ROTATE_180)
            mask = cv2.rotate(mask, cv2.cv2.ROTATE_180)
        H, W, _ = img.shape
        size = int(patch_size)

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
            #     lam = np.random.beta(1.0, 1.0)
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

        # if self.choice == 'train' and (H != int(patch_size) or W != int(patch_size)):
        #     x = 0 if H == int(patch_size) else np.random.randint(0, H-size)
        #     y = 0 if W == int(patch_size) else np.random.randint(0, W-size)
        #     img = img[x:x + size, y:y + size, :]
        #     mask = mask[x:x + size, y:y + size, :]
        elif self.choice == 'val':
            transformed = self.val_transform(image= img,mask=mask)
            img,mask= transformed['image'],transformed['mask']
            mask = mask[:, :, :1].permute(2, 0, 1).float() / 255
            # img =cv2.resize(img,(size,size))
            # mask = cv2.resize(mask,(size,size))
            # mask[mask < 127.5] = 0
            # mask[mask >= 127.5] = 255
        # img = img.astype('float') / 255.
        # mask = mask.astype('float') / 255.
        #return self.transform(img), self.tensor(mask[:, :, :1]), fname1.split('/')[-1]
        return img , mask, fname1.split('/')[-1]

    def aug(self, img, mask):
        # Resize the training data if necessary
        H, W, _ = img.shape
        if H < int(patch_size) or W < int(patch_size):
            m = int(patch_size) / min(H, W)
            img = cv2.resize(img, (int(H * m) + 1, int(W * m) + 1))
            mask = cv2.resize(mask, (int(H * m) + 1, int(W * m) + 1))
            mask[mask < 127.5] = 0
            mask[mask >= 127.5] = 255

#        Resize
        if random.random() < 0.5:
            H, W, C = img.shape
            if H * 0.9 > int(patch_size) and W * 0.9 > int(patch_size):
                r1, r2 = np.random.randint(90, 110) / 100., np.random.randint(90, 110) / 100.
            else:
                r1, r2 = np.random.randint(101, 110) / 100., np.random.randint(101, 110) / 100.
            img = cv2.resize(img, (int(H * r1), int(W * r2)))
            mask = cv2.resize(mask, (int(H * r1), int(W * r2)))
            mask[mask < 127.5] = 0
            mask[mask >= 127.5] = 255

        # Flip
        if random.random() < 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < 0.5:
            tmp = random.random()
            if tmp < 0.33:
                img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
                mask = cv2.rotate(mask, cv2.cv2.ROTATE_90_CLOCKWISE)
            elif tmp < 0.66:
                img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                mask = cv2.rotate(mask, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                img = cv2.rotate(img, cv2.cv2.ROTATE_180)
                mask = cv2.rotate(mask, cv2.cv2.ROTATE_180)

        # Noise
        if random.random() < 0.2:
            H, W, C = img.shape
            Nd = np.random.randint(10, 50) / 1000.
            # Nd = np.random.randint(30, 88) / 1000.
            Sd = 1 - Nd
            m = np.random.choice((0, 1, 2), size=(H, W, 1), p=[Nd / 2.0, Nd / 2.0, Sd])
            m = np.repeat(m, C, axis=2)
            # m[mask == 0] = 2
            img[m == 0] = 0
            img[m == 1] = 255
        if random.random() < 0.2:
            H, W, C = img.shape
            N = np.random.randint(10, 50) / 10. * np.random.normal(loc=0, scale=1, size=(H, W, 1))
            # N = np.random.randint(30, 88) / 10. * np.random.normal(loc=0, scale=1, size=(H, W, 1))
            N = np.repeat(N, C, axis=2)
            img = img.astype(np.int32)
            #img[mask == 255] = N[mask == 255] + img[mask == 255]
            img = N + img
            img[img > 255] = 255
            img = img.astype(np.uint8)

        return img, mask

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)

class GIID_Model(nn.Module):
    def __init__(self,backbone_arch='seresnext50',date='',params='',pretrained=True):
        super(GIID_Model, self).__init__()
        self.lr = 3e-4
        self.backbone_arch = backbone_arch
        if self.backbone_arch.startswith('se'):
            print(backbone_arch)
            self.networks = SCSEUnet(backbone_arch=backbone_arch,pretrained=pretrained) # seresnext50 , senet154
        elif self.backbone_arch.startswith('Upp'):
            print(backbone_arch)
            self.networks = seg.UnetPlusPlus(
                encoder_name="senet154",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7，se_resnet152
                encoder_weights='imagenet',  # use `imagenet` pretreined weights for encoder initialization
                decoder_attention_type='scse',
                in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=1,  # model output channels (number of classes in your dataset)
                activation='sigmoid'
            )
        else:
            print(backbone_arch)
            self.networks = seg.Linknet(
        encoder_name="senet154",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7，se_resnet152
        encoder_weights='imagenet',  # use `imagenet` pretreined weights for encoder initialization
        # decoder_attention_type='scse',
        in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
        activation= 'sigmoid'
    )
        self.gen = nn.DataParallel(self.networks).cuda().half()
        # self.gen_optimizer = optim.SGD(self.gen.parameters(),lr=1e-2,momentum=0.9,weight_decay=5e-4)
        self.gen_optimizer = optim.AdamW(self.gen.parameters(), lr=3e-4,weight_decay=5e-4, betas=(0.9, 0.999))
        # self.gen_optimizer = optim.AdamW(self.gen.parameters(), lr=1e-4, betas=(0.9, 0.999))
        # self.gen_scheduler =CosineAnnealingLR(self.gen_optimizer,T_max=80,eta_min=1e-6)
        self.gen_scheduler = CosineAnnealingWarmRestarts(self.gen_optimizer , T_0=3, T_mult=2, eta_min=1e-6, last_epoch=-1)
        self.save_dir = '../user_data/model_data_'+backbone_arch+'/'+date+params+'/'
        self.bce_loss = nn.BCELoss()
        # self.thx_loss = TverskyLoss()
        self.dice_loss = SoftDiceLoss()
        self.focal_loss = FocalLoss(mode='binary', alpha=0.25, gamma=2)
        self.LovaszLoss_fn=LovaszLoss(mode='multiclass',classes=[1])
        # DiceLoss
    def process(self, Ii, Mg):
        Mo = self(Ii)
        bce_loss = self.bce_loss(Mo.view(Mo.size(0), -1), Mg.view(Mg.size(0), -1))
        focal_loss  =self.focal_loss(Mo,Mg)
        # thx_loss = self.thx_loss(Mg,Mo)
        dice_loss = self.dice_loss(Mo,Mg)
        # lova_loss = 1 - self.LovaszLoss_fn(Mo,Mg)
        # 返回 F，D，L,B 格式
        return Mo, focal_loss,dice_loss,0,bce_loss

    def forward(self, Ii):
        return self.gen(Ii)

    def backward(self, gen_loss=None):
        if gen_loss:
            # scaler.scale(gen_loss).backward()
            # scaler.step( self.gen_optimizer)
            # scaler.update()
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

class ForgeryForensics():
    def __init__(self, fold=1,backbone_arch='seresnext50'):
        self.fold = fold
        self.test_num = 1
        self.batch_size =16
        self.cutmix_prob=0.5
        # self.train_npy = 'train_768.npy'
        # # self.train_npy ='train_And_s3_768_well6171_1.5.npy'
        # self.val_npy = 'val_768.npy'
        # self.train_file = np.load('../user_data/flist1/' + self.train_npy)
        # self.val_file = np.load('../user_data/flist1/' + self.val_npy)
        # self.train_file = np.concatenate([self.train_file,self.val_file[:500] ])
        # self.val_file = self.val_file[500:]
        # self.train_file = np.concatenate([self.train_file, self.val_file[1008:]])
        # self.train_file = np.concatenate([self.train_file, np.load('../user_data/' + 'train_And_s3_768_bad5157_1.3.npy')])
        # self.val_file = np.concatenate([self.val_file[:1008], np.load('../user_data/flist_add/' + 'val_erasing_10.npy')])
        self.train_npy = 'train_3600.npy'
        self.val_npy = 'val_400.npy'
        self.train_file = np.load('../user_data/flist_noDeco/' + self.train_npy)
        self.val_file = np.load('../user_data/flist_noDeco/' + self.val_npy)
        self.train_file = np.concatenate([self.train_file, np.load('../user_data/flist_add/' + 'train_s3data_1800.npy')[:1000]])
        self.train_file = np.concatenate([self.train_file, np.load('../user_data/flist_add/' + 'train_book_screen_1446.npy')[:1000]])
        # self.train_file = np.concatenate([self.train_file, np.load('../user_data/flist_add/' + 'val_s3data_200.npy')])
        # self.val_file = np.load('../user_data/flist_add/' + self.val_npy)
        # self.train_file = np.concatenate([self.train_file, np.load('../user_data/flist1/' + 'train_s3_768.npy')[::10]])
        # self.val_file = np.concatenate([self.val_file, np.load('../user_data/flist1/' + 'val_s3_768.npy')[:150]])
        # self.train_file = np.concatenate([self.train_file, np.load('../user_data/flist_add/' + 'train_erasing_90.npy')])
        # self.val_file = np.concatenate([self.val_file, np.load('../user_data/flist_add/' + 'val_erasing_10.npy')])
        self.train_num = len(self.train_file)
        self.val_num = len(self.val_file)
        print("train_num:%d,val_num:%d"%(self.train_num,self.val_num))
        train_dataset = GIID_Dataset(self.train_num, self.train_file, choice='train')
        val_dataset = GIID_Dataset(self.val_num, self.val_file, choice='val')
        print("loading model")
        self.giid_model = GIID_Model(backbone_arch=backbone_arch,date=date,params=params).cuda()
        print("loading successful")
        self.n_epochs = 100000000
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,pin_memory=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=12, shuffle=False, num_workers=4,pin_memory=True)

    def train(self):
        # with open('log_' + gpu_ids[0] + '.txt', 'a+') as f:
        #     f.write('\nTrain/Val with ' + self.train_npy + '/' + self.val_npy)
        cnt, gen_losses, f1, iou = 0, [], [], []
        train_f_losses,train_b_losses,train_d_losses,train_l_losses=[],[],[],[]
        val_f_losses,val_b_losses,val_l_losses,val_d_losses=[],[],[],[]
        tmp_epoch = 0
        # best_score = train_fold_best_val_record[self.fold]
        best_score = 0
        for epoch in range(1, self.n_epochs):
            f_avg =AverageMeter()
            d_avg = AverageMeter()
            l_avg = AverageMeter()
            b_avg = AverageMeter()
            for items in tqdm(self.train_loader):
                # break
                cnt += self.batch_size
                self.giid_model.train()
                ''' 半精度训练 '''
                Ii, Mg = (item.cuda() for item in items[:-1])

                Mo, f_loss,dice_loss,Lova_loss,bce_loss = self.giid_model.process(Ii, Mg)
                # loss = bce_loss
                loss = bce_loss+0.2*dice_loss
                # loss = 0.7*bce_loss+0.3*dice_loss+1*f_loss
                self.giid_model.backward(loss)
                gen_losses.append(loss.item())
                f_avg.update(f_loss.item() if f_loss!=0 else 0,Mo.size(0))
                d_avg.update(dice_loss.item()if dice_loss!=0 else 0,Mo.size(0))
                l_avg.update(Lova_loss.item()if Lova_loss!=0 else 0, Mo.size(0))
                b_avg.update(bce_loss.item()if bce_loss!=0 else 0,Mo.size(0))
                Mg, Mo = self.convert2(Mg), self.convert2(Mo)
                # break
                Mo[Mo < 127.5] = 0
                Mo[Mo >= 127.5] = 255
                for i in range(len(Ii)):
                    a, b = metric(Mo[i] / 255, Mg[i] / 255)
                    f1.append(a)
                    iou.append(b)
                if cnt % (self.batch_size*10)==0:
                    logging.info('epoch:%d,focal_loss:%5.4f,dice_loss:%5.4f,lova_loss:%5.4f,bce_loss:%5.4f' % (epoch,f_avg.avg,d_avg.avg,l_avg.avg,b_avg.avg)) #np.mean(gen_losses,axis=0)[1]
                if cnt % (self.batch_size*50)==0:
                    logging.info('Tra (%d/%d): Loss:%5.4f F1:%5.4f IOU:%5.4f SUM:%5.4f'
                      % (cnt, self.train_num, np.mean(gen_losses), np.mean(f1), np.mean(iou), np.mean(f1) + np.mean(iou)))
                # if cnt >= 10000:
            train_f_losses.append(f_avg.avg)
            train_l_losses.append(l_avg.avg)
            train_b_losses.append(b_avg.avg)
            train_d_losses.append(d_avg.avg)
            self.giid_model.lr_scheduler()
            val_gen_loss,val_f_loss,val_d_loss,val_l_loss,val_b_loss,val_f1, val_iou = self.val()
            val_b_losses.append(val_b_loss)
            val_d_losses.append(val_d_loss)
            val_f_losses.append(val_f_loss)
            val_l_losses.append(val_l_loss)
            logging.info('Val (%d/%d): Focal:%5.4f Dice:%5.4f Lova:%5.4f,bce:%5.4f F1:%5.4f IOU:%5.4f SUM:%5.4f'
                  % (cnt, self.train_num, val_f_loss,val_d_loss,val_l_loss,val_b_loss, val_f1, val_iou, val_f1 + val_iou))
            tmp_epoch = tmp_epoch + 1
            # self.giid_model.save('model_history_fold_%d' % self.fold + '/tmp_epoch_%03d/' % tmp_epoch)
            self.giid_model.save( 'model_history' + '/tmp_epoch_%03d_train%5.4f_val%5.4f_valLoss%5.4f/' % (tmp_epoch, np.mean(f1) + np.mean(iou),val_f1 + val_iou,val_gen_loss))
            # if np.mean(val_f1) + np.mean(val_iou) > best_score and tmp_epoch >= 3:
                # best_score = np.mean(val_f1) + np.mean(val_iou)
                #train_fold_best_val_record[self.fold] = best_score
                # np.save('train_fold_best_val_record.npy', train_fold_best_val_record)
                # self.giid_model.save('best_fold_%d/' % self.fold)
                # self.giid_model.save('best_model_%5.4f/' % best_score)
            # with open('log_' + gpu_ids[0] + '.txt', 'a+') as f:
            #     f.write('\n(%d/%d): Tra: G:%5.4f F1:%5.4f IOU:%5.4f SUM:%5.4f Val: G:%5.4f F1:%5.4f IOU:%5.4f SUM:%5.4f'
            #             % (cnt, self.train_num, np.mean(gen_losses), np.mean(f1), np.mean(iou), np.mean(f1) + np.mean(iou), val_gen_loss, val_f1, val_iou, val_f1 + val_iou))
            cnt, gen_losses, f1, iou = 0, [], [], []
            if tmp_epoch % 20==0:
                train_history = pd.DataFrame({
                    'focal_loss': train_f_losses,
                    'dice_loss': train_d_losses,
                    'Lovasz_loss':train_l_losses,
                    'bce_loss':train_b_losses
                })
                train_history.to_csv(os.path.join(self.giid_model.save_dir, 'train_%d.csv'%tmp_epoch))
                val_history = pd.DataFrame({
                    'focal_loss': val_f_losses,
                    'dice_loss': val_d_losses,
                    'Lovasz_loss': val_l_losses,
                    'bce_loss': val_b_losses
                })
                val_history.to_csv(os.path.join(self.giid_model.save_dir, 'val_%d.csv' % tmp_epoch))
            if (self.fold == 1 or self.fold == 2) and tmp_epoch >= 120:
                exit()
            if tmp_epoch >= 120:
                exit()

    def val(self):
        self.giid_model.eval()
        f1, iou, gen_losses = [], [], []
        f_avg = AverageMeter()
        d_avg = AverageMeter()
        l_avg = AverageMeter()
        b_avg = AverageMeter()
        # rm_and_make_dir('../user_data/res/val_fold%d_'%self.fold + gpu_ids[0] + '/')
        with torch.no_grad():
            for items in tqdm(self.val_loader):
                Ii, Mg = (item.cuda() for item in items[:-1])
                filename = items[-1]
                # print(filename)
                Mo, f_loss, dice_loss, Lova_loss,bce_loss= self.giid_model.process(Ii, Mg)
                # loss = bce_loss
                # loss = 3 * f_loss +  Lova_loss + 5 * dice_loss\
                loss = bce_loss + 0.2 * dice_loss
                # loss = 0.7*bce_loss+0.3*dice_loss+1*f_loss
                # loss = 3 * f_loss + dice_loss
                # loss = bce_loss + 0.6 * Lova_loss
                gen_losses.append(loss.item())
                f_avg.update(f_loss.item() if f_loss!=0 else 0,Mo.size(0))
                d_avg.update(dice_loss.item()if dice_loss!=0 else 0,Mo.size(0))
                l_avg.update(Lova_loss.item()if Lova_loss!=0 else 0, Mo.size(0))
                b_avg.update(bce_loss.item()if bce_loss!=0 else 0,Mo.size(0))
                Ii, Mg, Mo = self.convert1(Ii), self.convert2(Mg), self.convert2(Mo)
                N, H, W, _ = Mg.shape
                Mo[Mo < 127.5] = 0
                Mo[Mo >= 127.5] = 255
                for i in range(len(Ii)):
                    a, b = metric(Mo[i] / 255, Mg[i] / 255)
                    f1.append(a)
                    iou.append(b)

        return np.mean(gen_losses),f_avg.avg,d_avg.avg,l_avg.avg,b_avg.avg, np.mean(f1), np.mean(iou)
    def convert1(self, img):
        img = img * 127.5 + 127.5
        img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
        return img
    def convert2(self, x):
        x = x * 255.
        return x.permute(0, 2, 3, 1).cpu().detach().numpy()





import time
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str,default='train',help='train or test the model', choices=['train', 'test', 'val'])
    # parser.add_argument('--type', type=str, default='train', help='train or test the model')
    parser.add_argument('--func', type=int, default=0, help='test mode, one of [0, 1, 2] (\'decompose\', \'inference\', \'merge\')')
    parser.add_argument('--size_idx', type=int, default=0, help='index of [384, 512, 768, 1024] (input size)')
    parser.add_argument('--fold', type=int, default=1, help='one of [1, 2, 3, 4] (fold id)')
    parser.add_argument('--tta', type=int, default=1, help='one of [1, ..., 8] (TTA type)')
    args = parser.parse_args()
    # current_path = '../log/'
    TIME = time.asctime().split(' ')
    date = TIME[1]+'_'+TIME[2]+'_'
    logger = logging.getLogger(__name__)

    if args.type == 'train':
        params = 'AllImg_s3data_1k_bookscreen_1446_1k_size768_newDataAug_senet154_bce1_dice02'
        model = ForgeryForensics(args.fold,backbone_arch='senet154') # senet154,seresnext50,efficientnet-b4，Upp_se_resnet152
        os.makedirs(model.giid_model.save_dir,exist_ok=True)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.DEBUG,
            handlers=[
                logging.FileHandler(os.path.join(model.giid_model.save_dir, 'train.log')),
                logging.StreamHandler()
            ])
        # logging.info("senet154，分解图像，train:12020+280+3000, val:1008+20,Normalize:imagenet，optimizer:ADAM lr=1e-4,CAWR t0:5, tmul:2，CAWR t0:5, tmul:2,min=1e-7")
        logging.info(
            # "senet154，原始图像+s3，train:3600+1000+80, val:400+100+20,Normalize:imagenet,optimizer:ADAMw lr=3e-4,CAWR t0:3, tmul:2,min=1e-6,loss:focal3(alpha0.75,gamma2),LovaszLoss,dice5")  # senet154，unet efficient_b4
            "senet154，ALLimage3600,S3data1k,bookscreen1446_1k个，train:3600+1k+1k, val:400,patch_size=768,Copy_move0,Normalize:imagesnet,optimizer:ADAMw lr=3e-4,CAWR t0:3, tmul:2,,min=1e-6,loss,bce1,dice02,")  # senet154，unet efficient_b4
        model.train()