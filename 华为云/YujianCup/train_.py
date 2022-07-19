import os
import random
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch import optim
from transforms import gen_transform
from config import *
from utils import load_model, get_best_f1,rm_and_make_dir
import torch.cuda
import timm.scheduler as TScheduler
import logging

seed = 7777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


train_size = 224
resize = 224

autocast = torch.cuda.amp.autocast
scaler = torch.cuda.amp.GradScaler()

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,train_file,mode='train'):
        self.num= len(train_file)
        self.train_file = train_file
        # self.labels = label_file
        self.mode = mode
        self.transform = gen_transform(resize=train_size, mode=mode)
        logging.info(self.transform)
    def __getitem__(self, index):
        image,label= self.load_item(index)
        return image, label
    def __len__(self):
        return self.num
    def mixup(self,image,label):
        mixup_idx = random.randint(0, self.num - 1)
        image_path, mixup_target = self.train_file[mixup_idx]
        if train_args.LS > 0:
            mixup_label = np.array([abs(int(mixup_target) - train_args.LS)]).astype(np.float32)
        else:
            mixup_label = np.array([int(mixup_target)]).astype(np.float32)
        mixup_image = cv2.imread(image_path)[:, :, ::-1]
        mixup_image = cv2.resize(mixup_image, (resize, resize))
        mixup_image = mixup_image.astype(np.float32) / 255.
        transformed = self.transform(image=mixup_image)
        mixup_image = transformed['image']
        alpha = 0.2
        lam = np.random.beta(alpha, alpha)
        image = lam * image + (1 - lam) * mixup_image
        label = lam * label + (1 - lam) * mixup_label
        return image, label
    def load_item(self,index):
        if self.mode != 'val':
            image_path,target = self.train_file[index]
            if not os.path.exists(image_path):
                print(image_path)
            image = cv2.imread(image_path)[:,:,::-1]
            image = cv2.resize(image,(resize,resize))
            if train_args.LS > 0:
                label = np.array([abs(int(target) - train_args.LS)]).astype(np.float32)
            else:
                label = np.array([int(target)]).astype(np.float32)
            image = image.astype(np.float32)/255.
            transformed = self.transform(image=image)
            image = transformed['image']
            '''mixup'''
            if random.random()<train_args.MIXUP:
                image,label= self.mixup(image,label)
                label = label.astype(np.float32)
        else:
            image_path, target = self.train_file[index]
            image = cv2.imread(image_path)[:, :, ::-1]
            label = np.array([int(target)]).astype(np.float32)
            image = self.transform(image)
        return image,label

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

    print('Loading dataset....')
    train_file = np.load('./train_data/clean/clean.npy', allow_pickle=True)[::]  # 0
    train_file = np.concatenate([train_file, np.load('./train_data/adv/DeepFool.npy', allow_pickle=True)[1::2]])
    train_file = np.concatenate([train_file, np.load('./train_data/adv/Square.npy', allow_pickle=True)[1::2]])
    train_file = np.concatenate([train_file, np.load('./train_data/adv/Patch.npy', allow_pickle=True)[::2]])  #
    train_file = np.concatenate([train_file, np.load('./train_data/test/test1.npy', allow_pickle=True)])

    print('Loading dataset successfully')
    logging.info("TrainData num: %d " % len(train_file))
    TrainDataSet = MyDataset(train_file=train_file, mode='mix')
    trainloader = data.DataLoader(TrainDataSet, batch_size=args['batch_size'], shuffle=True, num_workers=4)


    val_file = np.load('train_data/test/test2.npy', allow_pickle=True)
    valDataSet =  MyDataset(train_file=val_file,mode='val')
    valloader = data.DataLoader(valDataSet,batch_size=256,shuffle=False,num_workers=4)

    # Model
    model = nn.DataParallel(load_model(arch,args['model_hyperparameters']))

    logging.info("Add MLP")
    in_features = model.module.fc.in_features
    out_features = model.module.fc.out_features
    model.module.fc= nn.Sequential(
            nn.Linear(in_features,3000),
            nn.ReLU(),
            nn.Linear(3000, out_features)
        )
    logger.info('Trainging :' + arch)
    logger.info(args)

    best_F1 = 0  # best test accuracy
    model = model.cuda()

    optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
                                                       **args['optimizer_hyperparameters'])
    if args['scheduler_name'] == 'CosineLRScheduler':
        n_iter_per_epoch = len(trainloader)
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

    for epoch in tqdm(range(args['epochs'])):
        train_pre,train_recall,train_f1 = train(trainloader, model, optimizer,scheduler,epoch)

        with torch.no_grad():
            F1,Pre,recall,acc= val(valloader,model)

        logging.info('Epoch: %d, train F1:%f, train Precision:%f,train Recall:%f' % (epoch+1,train_f1, train_pre,train_recall))
        logging.info( 'Epoch: %d, val F1 :%.4f, Pre:%.4f,recall:%.4f,acc:%.4f' % (epoch+1,F1,Pre,recall,acc))

        best_F1 = max(F1, best_F1)
        if F1>=0.75:
            save_checkpoint(model=model,arch=arch+'_epoch%d_TF1_%02.f_VF1_%02.f_P_%02.f_R_%02.f_acc_%02.f'%(epoch+1,train_f1*100,F1*100,Pre*100,recall*100,acc*100))
        if args['scheduler_name'] == 'CosineAnnealingWarmRestarts':
            scheduler.step()
    logging.info("Best acc:{}".format(best_F1))

def val(valloader, model):

    model.eval()
    labelss = None
    predicts = None
    with torch.no_grad():
        for (inputs, labels) in tqdm(valloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            predict = F.sigmoid(outputs)
            if labelss == None:
                labelss = labels
                predicts = predict
            else:
                labelss = torch.cat([labelss, labels])
                predicts = torch.cat([predicts, predict])

    BestF1,BestTP,BestFP,BestTN,BestFN,best_th = get_best_f1(predicts,labelss)
    torch.cuda.empty_cache()
    p =  BestTP / (BestTP + BestFP)
    r = BestTP / (BestTP + BestFN)
    acc = (BestTP+BestTN)/( BestTP + BestFP+BestTN+BestFN)
    return BestF1,p,r,acc

def train(trainloader, model, optimizer,scheduler,epoch):

    model.eval()
    labelss = None
    predicts = None
    num_steps= len(trainloader)
    model.train()
    for i,(inputs, labels) in enumerate(tqdm(trainloader)):
        inputs = inputs.to().cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)


        predict = F.sigmoid(outputs)
        if labelss == None:
            labelss = labels
            predicts = predict
        else:
            labelss = torch.cat([labelss, labels])
            predicts = torch.cat([predicts, predict])
        '''混合精度'''
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if args['scheduler_name'] == 'CosineLRScheduler':
            scheduler.step_update(epoch * num_steps + i)

    BestF1, BestTP, BestFP, BestTN, BestFN, best_th = get_best_f1(predicts, labelss)
    p =  BestTP / (BestTP + BestFP)
    r = BestTP / (BestTP + BestFN)
    return p,r,BestF1


def save_checkpoint(model, arch):
    filepath = os.path.join(current_path, arch + '.pth')
    torch.save(model.state_dict(), filepath)

import time
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--MIXUP',type=float,default=0.3,help='prob of using mixup')
    parser.add_argument('--LS', type=float, default=0.0, help='prob of using label Smoothing')
    parser.add_argument('--GPU', type=str, default='0,1', help='GPU')
    train_args= parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = train_args.GPU
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    args = args_resnet50
    TIME = time.asctime().split(' ')
    date = TIME[1]+'_'+TIME[2]+'_'
    params = 'Resnet50'
    current_path = "./model_data/"+date+params
    rm_and_make_dir(current_path)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(current_path, 'train.log')),
            logging.StreamHandler()
        ])
    logging.info(current_path)
    shutil.copy('config.py', os.path.join(current_path, 'config.py'))
    shutil.copy('train.py', os.path.join(current_path, 'train.py'))
    main()