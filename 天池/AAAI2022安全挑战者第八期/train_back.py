from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import random
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from config import args_resnet, args_densenet
from utils import load_model, AverageMeter, accuracy
import pandas as pd
# Use CUDA
use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


import logging



class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform,current_path):
        images = np.load(os.path.join(current_path,'data.npy'))
        labels = np.load(os.path.join(current_path,'label.npy'))
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

def main():

    for arch in ['resnet50', 'densenet121']:
        if arch == 'resnet50':
            args = args_resnet
            continue
        else:
            # continue
            args = args_densenet
        assert args['epochs'] <= 200
        # Data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = MyDataset(transform=transform_train,current_path=current_path)
        trainloader = data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=4)

        # Model
        model = load_model(arch,False,current_path)
        logger.info('Trainging :'+arch)
        logger.info(args)
        print('Trainging :'+arch)
        best_acc = 0  # best test accuracy
        optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
            **args['optimizer_hyperparameters'])
        if args['scheduler_name'] != None:
            scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
            **args['scheduler_hyperparameters'])
        model = model.cuda()
        # Train and val
        train_accs=[]
        test_accs = []
        train_losses=[]
        test_losses=[]
        for epoch in tqdm(range(args['epochs'])):
            train_loss, train_acc = train(trainloader, model, optimizer)
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            # test_loss,test_acc = test(testloader,model)
            # test_accs.append(test_acc)
            # test_losses.append(train_loss)
            # print('Train acc: {}; Test acc{}'.format(train_acc,test_acc))

            logging.info('train acc:%f, train loss:%f'%(train_acc, train_loss))
            logging.info(current_path[13:])
            # logging.info('Test acc:%f, Test loss:%f' % (test_acc, test_loss))
            # save model
            best_acc = max(train_acc, best_acc)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': train_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, arch=arch)
            if args['scheduler_name'] != None:
                scheduler.step()
        train_history=pd.DataFrame({
            'acc':train_accs,
            'loss':train_losses
        })
        train_history.to_csv(os.path.join(current_path,arch+'_train.csv'))
        logging.info("Best acc:{}".format(best_acc))

def test(testloader, model):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    for (inputs, soft_labels) in tqdm(testloader):
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = cross_entropy(outputs, soft_labels)
        acc = accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))
    return losses.avg, accs.avg
def train(trainloader, model, optimizer):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    # switch to train mode
    model.train()
    for (inputs, soft_labels) in tqdm(trainloader):
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = cross_entropy(outputs, soft_labels)
        acc = accuracy(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))
    return losses.avg, accs.avg

def save_checkpoint(state, arch):
    filepath = os.path.join(current_path,arch + '.pth.tar')
    torch.save(state, filepath)

if __name__ == '__main__':
    current_path='AllImage_FGSM003_030_1w_desnet_s_sp_p060_gs_white080_3w_mixup_all_1w'
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
    # if not os.path.exists(os.path.join(current_path,'config1.py')):
    shutil.copy('config.py',os.path.join(current_path,'config.py'))
    main()