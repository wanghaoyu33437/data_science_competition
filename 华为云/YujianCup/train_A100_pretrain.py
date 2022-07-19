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
    train_data_dir = './data/new_split_data/train'
    val_data_dir = './data/new_split_data/val'
    # test_data_dir = './data/test'
    train_dataset = MyDataset(train_data_dir, 'train',input_size=args['input_size'],resize=args['resize'])
    val_dataset = MyDataset(val_data_dir, 'val',input_size=args['input_size'],resize=args['resize'])
    image_datasets = {'train': train_dataset, 'val': val_dataset}

    print('The len of train dataset is %d'%len(train_dataset))
    print('The len of Val dataset is %d' % len(val_dataset))
    # test_datasets = MyDataset(test_data_dir,'val')
    # wrap your data and label into Tensor

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args['batch_size'],
                                                 shuffle=True,
                                                 num_workers=8,
                                                 ) for x in ['train', 'val']}


    # Define Model
    num_classes =  args['model_hyperparameters']['num_classes']
    args['model_hyperparameters']['num_classes'] = 128
    model = load_model(args['name'], args['model_hyperparameters'])
    dim_mlp = model.num_features
    model.head.fc = torch.nn.Sequential(
        torch.nn.Linear(dim_mlp,dim_mlp),
        torch.nn.ReLU(),
        model.head.fc
    )
    model.load_state_dict(torch.load('model_data/convnext_t_ssl_simclr/Convnext_T_epoch48_acc91.36324541284404_valAcc76.921875.pth',map_location='cpu'))
    num_features = model.num_features
    # for n, p in model.named_parameters():
    #     p.requires_grad_(False)
    model.head.fc = nn.Sequential(nn.Linear(num_features, 512),
                                  nn.ReLU(),
                                  nn.Linear(512,num_classes)
                                  )

    ema = EMA(model,args['ema_decay'])
    model = nn.DataParallel(model)
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
    class_weight = torch.FloatTensor([1.0,2.0,2.0,10.0]).cuda()
    LOSS = nn.CrossEntropyLoss(weight=class_weight,label_smoothing=args['LS'])
    for epoch in tqdm(range(args['epochs'])):
        if epoch ==args['warmup_epochs']:
            ema.register()
        train_acc,train_loss,train_adv_acc,train_adv_loss = train(model,dataloders['train'],optimizer,scheduler,LOSS,ema,epoch)
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
            ema.restore()
            logging.info(f'Val EMA  , Epoch:{epoch},acc: {val_ema_acc:.4f}, loss{val_ema_loss:.4f}')
            if log_writer:
                log_writer.add_scalar("Val/EMA_Acc", val_ema_acc, global_step=epoch)
                log_writer.add_scalar("Val/EMA_loss", val_ema_loss, global_step=epoch)
            # if val_acc > best_acc or val_ema_acc> best_acc:
            if val_acc < val_ema_acc:
                ema.apply_shadow()
                best_acc = val_ema_acc
                save_checkpoint(model, arch=arch + '_epoch%d_Tacc_%02.f_Vacc_%02.f_EMAVacc%02.f' % (
                epoch, train_acc * 100, val_acc * 100, val_ema_acc * 100))
                ema.restore()
            else:
                best_acc = val_acc
                save_checkpoint(model, arch=arch + '_epoch%d_Tacc_%02.f_Vacc_%02.f_EMAVacc%02.f' % (
                    epoch, train_acc * 100, val_acc * 100, val_ema_acc * 100))


        # Each epoch has a training and validation phase
        # lr = optimizer.
        else:
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model,arch=arch+'_epoch%d_Tacc_%02.f_Vacc_%02.f'%(epoch,train_acc*100,val_acc*100))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

def train(model,dataloader,optimizer,scheduler,LOSS,ema,epoch):
    accs = AverageMeter()
    losses = AverageMeter()
    adv_accs= AverageMeter()
    adv_losses = AverageMeter()
    model.eval()
    num_steps= len(dataloader)
    model.train()

    class_weight = torch.FloatTensor([1.0, 2.0, 2.0, 10.0]).cuda()
    for i,(inputs, labels) in enumerate(tqdm(dataloader)):
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        with autocast():
            if random.random() < args['mixup']:
                mix_inputs, lables_a, labels_b, lam = mixup(inputs, labels)
                outputs_A = model(mix_inputs)
                loss_A = mixup_criterion(LOSS, outputs_A, lables_a, labels_b, lam)
                # R_Drop
                outputs_B = model(mix_inputs)
                loss_B = mixup_criterion(LOSS, outputs_B, lables_a, labels_b, lam)
                kl_loss = compute_kl_loss(outputs_A,outputs_B)
                loss = loss_A+loss_B+0.3*kl_loss
            else:
                outputs_A = model(inputs)
                loss_A = LOSS(outputs_A, labels)
                outputs_B = model(inputs)
                loss_B = LOSS(outputs_B, labels)
                kl_loss = compute_kl_loss(outputs_A, outputs_B)
                loss = loss_A + loss_B + 0.3 * kl_loss
            if args['adv_train']:
                if epoch >= args['warmup_epochs']:
                    model.eval()
                    attack = torchattacks.__dict__[args['attack_method']](model=model, **args['attack_method_params'])
                    adv_image = attack(inputs,labels)
                    model.train()
                    adv_outputs = model(adv_image.cuda())
                    adv_loss = F.cross_entropy(adv_outputs, labels,weight=class_weight)
                    adv_acc = accuracy(adv_outputs,labels)
                    adv_accs.update(adv_acc[0].item(),len(labels))
                    adv_losses.update(adv_loss.item(),len(labels))
                    loss += adv_loss
        '''混合精度'''
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if epoch >= args['warmup_epochs']:
            if i%args['ema_step']==0:
                ema.update()
        # loss.backward()
        # optimizer.step()
        acc = accuracy(outputs_A,labels)
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
    parser.add_argument('--GPU', type=str, default='0,1', help='GPU')
    train_args= parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = train_args.GPU

    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

    # args = args_resnet50
    args= args_convnext_tiny
    TIME = time.asctime().split(' ')
    date = TIME[1]+'_'+TIME[2]+'_'
    params = f'{args["name"]}_PreTrain_noFreeze_Size{args["input_size"]}_bs{args["batch_size"]}_Split2_ema{args["ema_decay"]}_step{args["ema_step"]}_{"Mixup"if args["mixup"] else ""}_LS{args["LS"]}{"_adv" if args["adv_train"] else ""}/'
    current_path = "./model_data/"+date+params
    rm_and_make_dir(current_path)
    # logger = logging.getLogger(__name__)
    # log_writer = SummaryWriter(os.path.join(current_path, 'log'))
    log_writer =None
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(current_path, 'train.log')),
            logging.StreamHandler()
        ])
    logging.info(current_path)
    logging.info(args)
    shutil.copy('config.py', os.path.join(current_path, 'config.py'))
    shutil.copy('train_A100_pretrain.py', os.path.join(current_path, 'train.py'))
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


