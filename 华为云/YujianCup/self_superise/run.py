import argparse
import logging
import sys

import torch

sys.path.append('../')
import config
import torch.backends.cudnn as cudnn
from data_helper import MyDataset
import logging
import timm.scheduler as TScheduler
import random
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils import *

seed = 7777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
             self.device = torch.device('cpu')
        self.model = kwargs['model'].to(self.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        # self.writer = SummaryWriter(log_dir=os.path.join(self.args.project_name,'log'))
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader,val_loader):
        scaler = GradScaler()
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu.")
        best_Acc = 0
        for epoch_counter in range(self.args.epochs):
            num_steps= len(train_loader)
            self.model.train()
            accs_r1, accs_r5, losses = AverageMeter(), AverageMeter(), AverageMeter()
            for i,(images, _) in enumerate(tqdm(train_loader)):
                bs = len(images)
                images = torch.cat(images, dim=0)
                images = images.to(self.device)
                with autocast():
                    features = self.model(images)
                    # print(features.shape)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)
                losses.update(loss.item(),bs)
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                accs_r1.update(top1.item(),bs)
                accs_r5.update(top5.item(),bs)
                    # self.writer.add_scalar('loss', loss, global_step=n_iter)
                    # self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    # self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    # self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
            # self.model.eval()
            with torch.no_grad():
                val_acc,val_loss = self.val(val_loader)
            logging.info(f"Epoch: {epoch_counter}\t,Train  Loss: {losses.avg}\tTop1 accuracy: {accs_r1.avg}\tTop5 accuracy: {accs_r5.avg}")
            logging.info(f"Epoch: {epoch_counter}\t,Val Loss: {val_loss}\tTop1 accuracy: {val_acc}")
            if self.scheduler:
                self.scheduler.step_update(epoch_counter * num_steps + i)
            # logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
            # save model checkpoints
            if best_Acc<accs_r1.avg:
                best_Acc= accs_r1.avg
                arch = f'Convnext_T_epoch{epoch_counter}_acc{best_Acc}_valAcc{val_acc}'
                save_checkpoint(model=self.model,arch=arch)
    def val(self,dataloader):
        accs,losses=AverageMeter(),AverageMeter()
        for i, (images, _) in enumerate(tqdm(dataloader)):
            bs = len(images)
            images = torch.cat(images, dim=0)
            images = images.to(self.device)
            features = self.model(images)
            logits, labels = self.info_nce_loss(features)
            loss = self.criterion(logits, labels)
            losses.update(loss.item(), bs)
            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            accs.update(top1.item(), bs)
        return accs.avg,losses.avg
def save_checkpoint(model, arch):
    filepath = os.path.join(train_args.project_name, arch + '.pth')
    torch.save(model.state_dict(), filepath,_use_new_zipfile_serialization=False)
def main():
    cudnn.deterministic = True
    cudnn.benchmark = True
    train_data_dir = '../data/new_split_data/train'
    train_dataset = MyDataset(train_data_dir, 'ssl', input_size=384, resize=400)
    val_data_dir = '../data/new_split_data/val'
    val_dataset = MyDataset(val_data_dir, 'ssl', input_size=384, resize=400)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=train_args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=True)
    # args['model_hyperparameters']['pretrained']=False
    args['model_hyperparameters']['num_classes'] = 128
    model = load_model(args['name'], args['model_hyperparameters'])
    dim_mlp = model.num_features
    model.head.fc = torch.nn.Sequential(
        torch.nn.Linear(dim_mlp,dim_mlp),
        torch.nn.ReLU(),
        model.head.fc
    )

    optimizer = torch.optim.__dict__[args['optimizer_name']](model.parameters(),
                                                       **args['optimizer_hyperparameters'])
    if args['scheduler_name'] == 'CosineLRScheduler':
        n_iter_per_epoch = len(train_loader)
        num_steps = int(train_args.epochs * n_iter_per_epoch)
        warmup_steps = int(train_args.warmup_epochs * n_iter_per_epoch)
        decay_steps = int(args['decay_epochs'] * n_iter_per_epoch)
        scheduler = TScheduler.__dict__[args['scheduler_name']](optimizer,
                                                                t_initial=num_steps,
                                                                lr_min=args['min_lr'],
                                                                warmup_lr_init=args['warmup_lr'],
                                                                warmup_t=warmup_steps,
                                                                cycle_limit=1,
                                                                t_in_epochs=False,
                                                                )
    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=train_args)
    simclr.train(train_loader,val_loader)

import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument('--project_name', default='../model_data/convnext_t_ssl_simclr/', type=str)
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--warmup_epochs', default=5, type=int, metavar='N',  help='number of warmup_epochs')
    parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256), this is the total ')
    parser.add_argument('--out_dim', default=128, type=int,help='feature dimension (default: 128)')
    parser.add_argument('--log-every-n-steps', default=100, type=int, help='Log every n steps')
    parser.add_argument('--temperature', default=0.07, type=float,help='softmax temperature (default: 0.07)')
    parser.add_argument('--n-views', default=2, type=int, metavar='N', help='Number of views for contrastive learning training.')
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    train_args = parser.parse_args()
    args = config.args_convnext_tiny
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(train_args.project_name, 'train.log')),
            logging.StreamHandler()
        ])
    os.makedirs(train_args.project_name, exist_ok=True)
    os.makedirs(os.path.join(train_args.project_name, 'log'), exist_ok=True)
    logging.info(train_args)
    logging.info(args)
    main()
