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
from data_helper import MyDataset
from utils import load_model,accuracy,AverageMeter
import argparse
from trick import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str,default='./model_data/convnext_t_ssl_simclr/Convnext_T_epoch10_acc85.19567087155963.pth',help='prob of using mixup')
    parser.add_argument('--LS', type=float, default=0.0, help='prob of using label Smoothing')
    parser.add_argument('--GPU', type=str, default='0,1', help='GPU')
    os.environ['CUDA_VISIBLE_DEVICE']='1'
    train_args= parser.parse_args()
    test_data_dir = './data/new_split_data/val'
    test_datasets = MyDataset(test_data_dir, 'val',input_size=384,resize=400)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=128, num_workers=4)

    args = args_convnext_tiny
    arch = args['name']

    args['model_hyperparameters']['num_classes'] = 0
    model = load_model(arch, args['model_hyperparameters'])
    model.load_state_dict(torch.load(train_args.path,map_location='cpu'))
    model.head.fc = nn.Sequential(nn.Linear(768,4))
    model = nn.DataParallel(model).cuda()

    accs = AverageMeter()
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_dataloader)):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            acc = accuracy(outputs, labels)
            accs.update(acc[0].item(), len(labels))
            losses.update(loss.item(), len(labels))
    print(f'acc:{accs.avg} loss:{losses.avg}')