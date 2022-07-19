import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from self_superise.ssl_utils import save_config_file, save_checkpoint
from utils import *
torch.manual_seed(0)


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

    def train(self, train_loader):

        scaler = GradScaler()

        # save config file
        # save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        accs_r1,accs_r5,losses = AverageMeter(),AverageMeter(),AverageMeter()
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu.")

        for epoch_counter in range(self.args.epochs):
            num_steps= len(train_loader)
            for i,(images, _) in enumerate(tqdm(train_loader)):
                bs = len(images)
                images = torch.cat(images, dim=0)

                images = images.to(self.device)

                with autocast():
                    features = self.model(images)
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

                n_iter += 1
            if (epoch_counter * num_steps + i)%20 ==0:
                logging.info(f"Epoch: {epoch_counter}\tStep :{i}, Loss: {losses.avg}\tTop1 accuracy: {accs_r1.avg}\tTop5 accuracy: {accs_r5.avg}")
            # warmup for the first 10 epochs
            if self.scheduler:
                self.scheduler.step_update(epoch_counter * num_steps + i)

            # logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
        }, is_best=False, filename=os.path.join(self.args.project_name, checkpoint_name))
