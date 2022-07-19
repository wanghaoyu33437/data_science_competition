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
from data_helper import MyDataset
from utils import load_model,accuracy,AverageMeter
torch
# use gpu or not
use_gpu = torch.cuda.is_available()


def main():

    args = args_resnet50
    arch = args['name']

    print('Loading dataset....')
    data_dir = './data'
    train_data_dir = './data/train'
    val_data_dir = './data/val'
    test_data_dir = './data/test'
    train_dataset = MyDataset(train_data_dir, 'train')
    val_dataset = MyDataset(val_data_dir, 'val')
    image_datasets = {'train': train_dataset, 'val': val_dataset}
    test_datasets = MyDataset(test_data_dir,'val')
    # wrap your data and label into Tensor

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args['batch_size'],
                                                 shuffle=True,
                                                 num_workers=6,
                                                 ) for x in ['train', 'val']}

    test_dataloader = torch.utils.data.DataLoader(test_datasets,batch_size=64,num_workers=4)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}



    start_time = time.time()
    # Define Model
    model = nn.DataParallel(load_model(arch, args['model_hyperparameters']))
    if use_gpu :
        model = model.cuda()

    optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
                                                       **args['optimizer_hyperparameters'])
    if args['scheduler_name'] == 'CosineLRScheduler':
        n_iter_per_epoch = len(dataloders['train'])
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
    best_acc = 0.0
    for epoch in tqdm(range(args['epochs'])):
        train_acc,train_loss = train(model,dataloders['train'],optimizer,scheduler,epoch)
        with torch.no_grad():
            val_acc, val_loss = val(model,dataloders['val'])
        # Each epoch has a training and validation phase
        lr = optimizer
        logging.info(f'Train, Epoch:{epoch},acc: {train_acc:.4f}, loss{train_loss:.4f}')
        logging.info(f'Val  , Epoch:{epoch},acc: {val_acc:.4f}, loss{val_loss:.4f}')
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model,arch=arch+'_epoch%d_Tacc_%02.f_Vacc_%02.f'%(epoch,train_acc*100,val_acc*100))

    logging.info('Best val Acc: {:4f}'.format(best_acc))

def train(model,dataloader,optimizer,scheduler,epoch):
    accs = AverageMeter()
    losses = AverageMeter()
    model.eval()
    num_steps= len(dataloader)
    model.train()
    for i,(inputs, labels) in enumerate(tqdm(dataloader)):
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)

        '''混合精度'''
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        acc = accuracy(outputs,labels)
        accs.update(acc[0].item(),len(labels))
        losses.update(loss.item(),len(labels))
        if args['scheduler_name'] == 'CosineLRScheduler':
            scheduler.step_update(epoch * num_steps + i)
        if args['scheduler_name'] == 'CosineAnnealingWarmRestarts':
            scheduler.step_update(epoch + i/num_steps )
    return accs.avg,losses.avg
def val(model,dataloader):
    accs = AverageMeter()
    losses = AverageMeter()
    model.eval()
    for i,(inputs, labels) in enumerate(tqdm(dataloader)):
        inputs = inputs.cuda()

        labels = labels.cuda()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc = accuracy(outputs,labels)
        accs.update(acc[0].item(),len(labels))
        losses.update(loss.item(),len(labels))
    return accs.avg,losses.avg
def save_checkpoint(model, arch):
    filepath = os.path.join(current_path, arch + '.pth')
    torch.save(model.module.state_dict() if model.module else model.state_dict(), filepath,_use_new_zipfile_serialization=False)
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

# model_ft = models.resnet50(pretrained=True, progress=False)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, len(train_dataset.classes))

# if use_gpu:
#     model_ft = model_ft.cuda()
#
#     # define loss function
# lossfunc = nn.CrossEntropyLoss()
#
# # setting optimizer and trainable parameters
# #   params = model_ft.parameters()
# # list(model_ft.fc.parameters())+list(model_ft.layer4.parameters())
# # params = list(model_ft.fc.parameters())+list( model_ft.parameters())
# params = list(model_ft.fc.parameters())
# optimizer_ft = optim.SGD(params, lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#
# model_ft = train_model(model=model_ft,
#                        lossfunc=lossfunc,
#                        optimizer=optimizer_ft,
#                        scheduler=exp_lr_scheduler,
#                        num_epochs=50)
#
# torch.save(model_ft.state_dict(), './model/model.pth', _use_new_zipfile_serialization=False)


