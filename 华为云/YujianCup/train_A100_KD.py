import os
import random
import shutil

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch import optim
from config import *
import torch.cuda
import timm.scheduler as TScheduler
import logging
from data_helper import MyDataset
from utils import load_model,accuracy,AverageMeter,rm_and_make_dir
import csv
from loss.ce_loss import  CELoss
from trick import *
import torchattacks
from torch.utils.tensorboard import SummaryWriter
# use gpu or not
use_gpu = torch.cuda.is_available()

seed = 7777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def main():
    arch = args['name']
    print('Loading dataset....')
    train_data_dir = './data/new_data/train'
    val_data_dir = './data/new_data/val'
    # test_data_dir = './data/test'
    train_dataset = MyDataset(train_data_dir, 'train',resize=args['input_size'])
    val_dataset = MyDataset(val_data_dir, 'val',resize=args['input_size'])
    image_datasets = {'train': train_dataset, 'val': val_dataset}
    # test_datasets = MyDataset(test_data_dir,'val')
    # wrap your data and label into Tensor

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args['batch_size'],
                                                 shuffle=True,
                                                 num_workers=6,
                                                 ) for x in ['train', 'val']}


    # Define Model
    model = load_model(arch, args['model_hyperparameters'])
    model = nn.Sequential(
        F_interpolate(384),
        NormalizeLayer([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        model
    )

    teacher_model = load_model(arch, args['model_hyperparameters'])
    teacher_model.load_state_dict(torch.load('model_data/Jul_4_test_convnext_tiny_Size1024_bs16_AllData_CLWU_ema0.999_step1__LS1.0_adv/convnext_tiny_acc8826.pth'))
    teacher_model.eval()
    teacher_model = nn.Sequential(
        F_interpolate(1024),
        NormalizeLayer([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        teacher_model
    )
    for n,p in teacher_model.named_parameters():
        p.requires_grad_(False)

    ema = EMA(model,args['ema_decay'])
    model = nn.DataParallel(model)

    if use_gpu :
        teacher_model = teacher_model.cuda()
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
    LOSS = CELoss(label_smooth=args['LS'],class_num=4)
    best_acc = 0.0
    for epoch in tqdm(range(args['epochs'])):
        if epoch ==args['warmup_epochs']:
            ema.register()
        train_acc,train_loss,train_adv_acc,train_adv_loss = train(model,teacher_model,dataloders['train'],optimizer,scheduler,LOSS,ema,epoch)
        with torch.no_grad():
            val_acc, val_loss = val(model,dataloders['val'])
        logging.info(f'Train, Epoch:{epoch},acc: {train_acc:.4f}, loss{train_loss:.4f}, adv acc: {train_adv_acc:.4f}, adv loss{train_adv_loss:.4f}')
        logging.info(f'Val  , Epoch:{epoch},acc: {val_acc:.4f}, loss{val_loss:.4f}')
        if log_writer:
            log_writer.add_scalar("Val/Acc", val_acc, global_step=epoch)
            log_writer.add_scalar("Val/loss", val_loss, global_step=epoch)
        # train_log_writer.writerow([epoch, train_acc, train_loss])
        # train_log_file.flush()
        # val_log_writer.writerow([epoch, val_acc, val_loss])
        # val_log_file.flush()

        if epoch >= args['warmup_epochs']:
            ema.apply_shadow()
            with torch.no_grad():
                val_ema_acc, val_ema_loss = val(model, dataloders['val'])
            logging.info(f'Val EMA  , Epoch:{epoch},acc: {val_ema_acc:.4f}, loss{val_ema_loss:.4f}')
            if log_writer:
                log_writer.add_scalar("Val/EMA_Acc", val_ema_acc, global_step=epoch)
                log_writer.add_scalar("Val/EMA_loss", val_ema_loss, global_step=epoch)
            if val_ema_acc > best_acc:
                best_acc = val_ema_acc
                save_checkpoint(model, arch=arch + '_epoch%d_Tacc_%02.f_Vacc_%02.f_EMAVacc%02.f' % (
                epoch, train_acc * 100, val_acc * 100, val_ema_acc * 100))
            ema.restore()
        # Each epoch has a training and validation phase
        # lr = optimizer.
        else:
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model,arch=arch+'_epoch%d_Tacc_%02.f_Vacc_%02.f'%(epoch,train_acc*100,val_acc*100))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

def train(model,teacher_model,dataloader,optimizer,scheduler,LOSS,ema,epoch):
    accs = AverageMeter()
    losses = AverageMeter()
    adv_accs= AverageMeter()
    adv_losses = AverageMeter()
    model.eval()
    num_steps= len(dataloader)
    model.train()

    for i,(inputs, labels) in enumerate(tqdm(dataloader)):
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        if args['mixup']:
            mix_inputs,lables_a,labels_b,lam = mixup(inputs,labels)
        with autocast():
            if args['mixup']:
                outputs = model(mix_inputs)
                loss = mixup_criterion(LOSS,outputs,lables_a,labels_b,lam)
            else:
                with torch.no_grad():
                    teacher_labels = teacher_model(inputs)
                    # print(accuracy(teacher_labels,labels))
                outputs = model(inputs)
                loss = LOSS(outputs, labels,soft_label=teacher_labels)
                # loss = LOSS(outputs, labels)
            if args['adv_train']:
                model.eval()
                attack = torchattacks.__dict__[args['attack_method']](model=model, **args['attack_method_params'])
                adv_image = attack(inputs,labels)
                model.train()
                adv_outputs = model(adv_image.cuda())
                # with torch.no_grad():
                    # adv_teacher_labels = teacher_model(adv_image.cuda())
                    # print(accuracy(adv_teacher_labels, labels))
                # adv_loss = LOSS(adv_outputs, labels)
                adv_loss = F.cross_entropy(adv_outputs,labels)
                adv_acc = accuracy(adv_outputs,labels)
                adv_accs.update(adv_acc[0].item(),len(labels))
                adv_losses.update(adv_loss.item(),len(labels))
                loss += adv_loss
        '''混合精度'''
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if epoch>=5:
            if i%args['ema_step']==0:
                ema.update()
        # loss.backward()
        # optimizer.step()
        acc = accuracy(outputs,labels)
        accs.update(acc[0].item(),len(labels))
        losses.update(loss.item(),len(labels))
        if log_writer:
            log_writer.add_scalar("Train/Acc", acc[0].item(), global_step=(epoch * num_steps + i))
            log_writer.add_scalar("Train/loss", loss.item(), global_step=(epoch * num_steps + i))
        if args['scheduler_name'] == 'CosineLRScheduler':
            scheduler.step_update(epoch * num_steps + i)
        if args['scheduler_name'] == 'CosineAnnealingWarmRestarts':
            scheduler.step(epoch + i/num_steps )
    return accs.avg,losses.avg,adv_accs.avg,adv_losses.avg
def val(model,dataloader):
    accs = AverageMeter()
    losses = AverageMeter()
    model.eval()
    for i,(inputs, labels) in enumerate(tqdm(dataloader)):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(F.interpolate(inputs,(384,384)))
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
    parser.add_argument('--GPU', type=str, default='0,1', help='GPU')
    train_args= parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = train_args.GPU

    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

    # args = args_resnet50
    args= args_convnext_tiny
    TIME = time.asctime().split(' ')
    date = TIME[1]+'_'+TIME[3]+'_'
    params = f'{args["name"]}_Size{args["input_size"]}_bs{args["batch_size"]}_AllData_CLWU_ema{args["ema_decay"]}_step{args["ema_step"]}{"_Mixup"if args["mixup"] else ""}_LS{args["LS"]*10}{"_adv" if args["adv_train"] else ""}_KD/'
    current_path = "./model_data/"+date+params
    rm_and_make_dir(current_path)
    # logger = logging.getLogger(__name__)
    # log_writer = SummaryWriter(os.path.join(current_path, 'log'))
    log_writer =None
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(current_path, 'train.log')),
            logging.StreamHandler()
        ])
    logging.info(current_path)
    logging.info(args)
    shutil.copy('config.py', os.path.join(current_path, 'config.py'))
    shutil.copy('train_A100.py', os.path.join(current_path, 'train.py'))
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


