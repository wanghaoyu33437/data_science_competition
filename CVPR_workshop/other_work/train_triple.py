from __future__ import print_function
import os
# from train import MyDataset
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import random
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from transform import gen_transform
import torchvision
import torchvision.datasets as datasets
from config import *

from utils import AverageMeter,load_model, accuracy,rm_and_make_dir
import pandas as pd
from TSNE import *
from optimizer.convnext_optim_factory import create_optimizer
import timm.scheduler as TScheduler

seed = 7777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
import torch.cuda
torch.backends.cudnn.deterministic = True
import logging


train_size = 224

scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast

def get_label(path):
    if 'JPEG'in path:
        return 0
    elif 'adv_005' in path:
        return 1
    elif 'deepfool' in path:
        return 2
    elif 'patch' in path:
        return 3
    elif 'square' in path:
        return 4
    else:
        return 5
class TripleDataset(torch.utils.data.Dataset):
    def __init__(self,positive_file,negative_file,mode='mix',anchor='normal',normal_num=0):
        self.positive_num= len(positive_file)
        self.negative_num= len(negative_file)
        self.positive_file = positive_file
        self.negative_file = negative_file
        self.mode = mode
        self.anchor = anchor
        self.normal_num=normal_num
        self.transform = gen_transform(resize=train_size, mode=mode)
        logging.info(self.transform)
    def __getitem__(self, index):
        return self.load_item(index)
    def __len__(self):
        return self.positive_num
    def load_item(self,index):
        anchor_path,target = self.positive_file[index]
        if not os.path.exists(anchor_path):
            print(anchor_path)
        anchor = cv2.imread(anchor_path)[:,:,::-1]
        anchor = cv2.resize(anchor,(train_size,train_size))
        anchor_label = np.array([get_label(anchor_path)]).astype(np.float32)

        positive_idx = random.randint(0, self.positive_num - 1)
        positive_path,positive_target = self.positive_file[positive_idx]
        positive = cv2.imread(positive_path)[:,:,::-1]
        positive = cv2.resize(positive,(train_size,train_size))
        positive_label =  np.array([get_label(positive_path)]).astype(np.float32)

        negative_idx = random.randint(0, self.negative_num - 1)
        negative_path, negative_target = self.negative_file[negative_idx]
        # negative_label = np.zeros([2]).astype(np.float32)
        negative = cv2.imread(negative_path)[:, :, ::-1]
        negative = cv2.resize(negative, (train_size, train_size))
        negative_label =  np.array([get_label(negative_path)]).astype(np.float32)
        # negative_label[int(negative_target)] = 1
        ''' 
        以adv样本为anchor
        '''
        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative =  self.transform(negative)
        '''mixup'''
        return anchor,positive,negative,anchor_label,positive_label,negative_label
    def mixup(self,image,label):
        mixup_idx = random.randint(0, self.num-1)
        image_path, mixup_target = self.train_file[mixup_idx]
        mixup_label = np.zeros([2]).astype(np.float32)
        mixup_label[int(mixup_target)] = 1
        mixup_image = cv2.imread(image_path)[:, :, ::-1]
        # mixup_image = cv2.resize(mixup_image,(train_size,train_size))
        # Select a random number from the given beta distribution
        # Mixup the images accordingly
        transformed = self.transform(image=mixup_image)
        mixup_image = transformed['image']
        alpha = 0.2
        lam = np.random.beta(alpha, alpha)
        image = lam * image + (1 - lam) * mixup_image
        label = lam * label + (1 - lam) * mixup_label
        return image, label


def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

def labelSmoothing(target,smoothing,classes=20,dim=-1):
    confidence = 1.0 - smoothing
    true_dist=torch.zeros_like(torch.tensor(np.random.random((target.shape[0], classes))))
    true_dist.fill_(smoothing / (classes - 1))
    true_dist.scatter_(1, target.detach().cpu().unsqueeze(1), confidence)
    return true_dist
def main():
    arch = args['name']
    if args['batch_size'] > 256:
        # force the batch_size to 256, and scaling the lr
        args['optimizer_hyperparameters']['lr'] *= 256 / args['batch_size']
        args['batch_size'] = 256
    # Data
    print('Loading dataset....')
    loader_list = []

    clean_file = np.load('./Phase2_traindata/CleanImage/clean.npy',allow_pickle=True)
    adv5 = np.load('./Phase2_traindata/ADVImage/adv5.npy', allow_pickle=True)

    adv_patch = np.load('./Phase2_traindata/ADVImage/advPatch.npy', allow_pickle=True)
    adv_deepfool = np.load('./Phase2_traindata/ADVImage/adv_deepfool01.npy', allow_pickle=True)
    adv_square = np.load('./Phase2_traindata/ADVImage/adv_square.npy', allow_pickle=True)

    adv = np.concatenate([adv5[::2],adv_patch[1::2],adv_deepfool[::2],adv_square[1::2]])
    TrainDataSet = TripleDataset(positive_file=adv,negative_file=clean_file, mode='mix',)
    trainloader = data.DataLoader(TrainDataSet,batch_size = 128,shuffle=False,num_workers=4)
    # ValDataSet = TripleDataset(positive_file=adv5[20000::],
    #                               negative_file=clean_file[20000::], mode='val',
    #                              )
    # loader_list.append(['adv5',data.DataLoader(TrainDataSet, batch_size=args['batch_size'], shuffle=True, num_workers=6),
    #                     data.DataLoader(ValDataSet, batch_size=args['batch_size'], shuffle=False,num_workers=4)])
    #
    #
    # TrainDataSet = TripleDataset(positive_file=adv_patch[:20000:],
    #                               negative_file=clean_file[:20000:], mode='mix',
    #                              )
    # ValDataSet = TripleDataset(positive_file=adv_patch[20000::],
    #                               negative_file=clean_file[20000::], mode='val',
    #                              )
    # loader_list.append(['patch',data.DataLoader(TrainDataSet, batch_size=args['batch_size'], shuffle=True, num_workers=6),
    #                     data.DataLoader(ValDataSet, batch_size=args['batch_size'], shuffle=False,num_workers=4)])
    #
    # TrainDataSet = TripleDataset(positive_file=adv_deepfool[:20000:],
    #                              negative_file=clean_file[:20000:], mode='mix',
    #                              )
    # ValDataSet = TripleDataset(positive_file=adv_deepfool[20000::],
    #                            negative_file=clean_file[20000::], mode='val',
    #                            )
    # loader_list.append(['deepfool',data.DataLoader(TrainDataSet, batch_size=args['batch_size'], shuffle=True, num_workers=6),
    #                     data.DataLoader(ValDataSet, batch_size=args['batch_size'], shuffle=False,num_workers=4)])
    #
    #
    # TrainDataSet = TripleDataset(positive_file=adv_square[:20000:],
    #                              negative_file=clean_file[:20000:], mode='mix',
    #                              )
    # ValDataSet = TripleDataset(positive_file=adv_square[20000::],
    #                            negative_file=clean_file[20000::], mode='val',
    #                            )
    # loader_list.append(['square',data.DataLoader(TrainDataSet, batch_size=args['batch_size'], shuffle=True, num_workers=6),
    #                     data.DataLoader(ValDataSet, batch_size=args['batch_size'], shuffle=False,num_workers=4)])
    from train_binary import MyDataset
    val_file = np.load('Phase2_traindata/track2_test2_adv373_5k.npy', allow_pickle=True)
    valDataSet =  MyDataset(train_file=val_file,mode='val',normal_num=len(val_file))
    valloader = torch.utils.data.DataLoader(valDataSet,batch_size=128,shuffle=False,num_workers=4)

    loader_list.append(['adv',trainloader,valloader])

    print('Loading dataset successfully')

    # Model
    model = nn.DataParallel(load_model(arch, args['model_hyperparameters']))
    model.module.fc = nn.Sequential()
    if args['resume'] ==True:
        print('Load preTrain weight')
        model.load_state_dict(torch.load(args['resume_path']))

    logger.info('Trainging :' + arch)
    logger.info(args)

    optimizer = create_optimizer(args, model)

    model = model.cuda()
    if args['scheduler_name'] == 'CosineLRScheduler':
        n_iter_per_epoch = len(loader_list[0][0])
        num_steps = int(args['epochs'] * n_iter_per_epoch)
        warmup_steps = int(args['warmup_epochs'] * n_iter_per_epoch)
        decay_steps = int(args['decay_epochs'] * n_iter_per_epoch)
        scheduler = TScheduler.__dict__[args['scheduler_name']](optimizer,
                                                                t_initial=num_steps,
                                                                lr_min=args['min_lr'],
                                                                warmup_lr_init=args['warmup_lr'],
                                                                warmup_t=warmup_steps,
                                                                cycle_limit=1,
                                                                t_in_epochs=False,
                                                                )
    else:
        scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
                                                                              **args['scheduler_hyperparameters'])
    model = model.cuda()
    train_epoch_loss = []
    val_epoch_loss = []
    for epoch in tqdm(range(args['epochs'])):
        for name,train_loader,val_loader, in loader_list:
            if epoch == 0:
                with torch.no_grad():
                    val(val_loader,model,epoch,name)
            train_loss= train(train_loader, model, optimizer,scheduler,epoch)
            with torch.no_grad():
                val_loss = val(val_loader, model,epoch+1,name)
            logging.info('Epoch: %d, train_triple_loss:%f,val_loss:%f' % (epoch+1,train_loss,val_loss))
        if args['scheduler_name'] == 'CosineAnnealingWarmRestarts':
            scheduler.step()
            # logging.info('Epoch: %d, val F1 :%f, TP:%d,FP:%d' % (epoch + 1, F1, TP, FP))
        # logging.info('Test acc:%f, Test loss:%f' % (test_acc, test_loss))
        # save model
        # best_acc = max(train_acc, best_acc)
        # if train_acc>=75:
        save_checkpoint(model=model, arch=arch + '_epoch%d_Tloss%.2f_Vloss%.2f' % (epoch + 1, train_loss, val_loss))

def train(train_loader, model, optimizer,scheduler,epoch):
    loss_tris = AverageMeter()
    model.eval()
    model.train()
    triplet_loss = nn.TripletMarginLoss(margin=3)
    num_steps = len(train_loader)
    for i,(anchor,positive,negative,_,_,_) in enumerate(tqdm(train_loader)):
        batch_size = len(anchor)
        anchor = anchor.cuda()# .half()
        positive= positive.cuda()
        negative = negative.cuda()
        inputs = torch.cat([anchor,positive,negative])
        # labels = torch.cat([label1,label2,label3])
        # labels  =labels.view(-1,labels.shape[-1])/
        # labels = labels.cuda()
        # targets = labels.argmax(dim=1)
        optimizer.zero_grad()
        with autocast():
            feature = model(inputs)
            anchor_f,positive_f,negative_f= feature[:batch_size],feature[batch_size:batch_size*2],feature[batch_size*2:]
            loss_tri = triplet_loss(anchor=anchor_f,positive=positive_f,negative=negative_f)
        loss = loss_tri
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if args['scheduler_name']=='CosineLRScheduler':
            scheduler.step(epoch*num_steps+i)
        loss_tris.update(loss_tri.item(), inputs.size(0))
    return loss_tris.avg

def val(valloader, model,epoch,name):
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)  # 调用TSNE
    model.eval()
    feature_all= None
    labels_all =None
    losses = AverageMeter()
    triplet_loss = nn.TripletMarginLoss(margin=3)
    with torch.no_grad():
        for inputs,lables in tqdm(valloader):
            batch_size = inputs.shape[0]
            inputs = inputs.cuda()
            feature = model(inputs)
            if feature_all is None:
                labels_all = lables
                feature_all = feature.detach().clone().cpu()
            else:
                labels_all = torch.cat([labels_all, lables])
                feature_all= torch.cat([feature_all,feature.detach().clone().cpu()])
            # losses.update(loss.item(),batch_size)
    result_2D = tsne_2D.fit_transform(feature_all)
    fig = plot_embedding_2D(result_2D,labels_all, '',64)
    os.makedirs(os.path.join(current_path,'fig',str(epoch)),exist_ok=True)
    fig.savefig(os.path.join(current_path,'fig',str(epoch),'tsne_%s_epoch_%d.png'%(name,epoch)))
    return losses.avg

def save_checkpoint(model, arch):
    filepath = os.path.join(current_path, arch + '.pth')
    torch.save(model.state_dict(), filepath)

import time
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
    parser.add_argument('--MIXUP',type=float,default=0.,help='prob of using mixup')
    parser.add_argument('--LS', type=float, default=0, help='prob of using label Smoothing')
    train_args= parser.parse_args()
    args = args_resnet50
    # args = args_convnext_base#args_swin_tiny_patch4_window7_224#,args_swin_s3_tiny_224,args_convnext_tiny_hnf
    TIME = time.asctime().split(' ')
    date = TIME[1]+'_'+TIME[2]+'_'
    params = 'Resnet50_Triple_clean_adv5_deepfool_Patch_squarer_resize224_noNormalize'
    # current_path = "./Phase2_model_data/"+date+params
    # rm_and_make_dir(current_path)
    # logger = logging.getLogger(__name__)
    # logging.basicConfig(
    #     format='[%(asctime)s] - %(message)s',
    #     datefmt='%Y/%m/%d %H:%M:%S',
    #     level=logging.DEBUG,
    #     handlers=[
    #         logging.FileHandler(os.path.join(current_path, 'train.log')),
    #         logging.StreamHandler()
    #     ])
    # logging.info(current_path)
    # # logging.info('数据组成：clean,all,corruption_prob0.2,adv4，patch_1/2')
    # # if not os.path.exists(os.path.join(current_path,'config1.py')):
    # shutil.copy('config.py', os.path.join(current_path, 'config.py'))
    # shutil.copy('train_triple.py', os.path.join(current_path, 'train_triple.py'))
    # main()