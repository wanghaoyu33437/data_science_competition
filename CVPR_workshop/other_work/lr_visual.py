import os
import glob
from tqdm import tqdm
import torch
from torchvision import models
import matplotlib.pyplot as plt
#画出余弦学习率的变化规律
def visulize_cosine_lr(net,max_epoch,optimizer,lr_scheduler,iters):
    plt.figure()
    cur_lr_list = []
    cur_lr = optimizer.param_groups[-1]['lr']
    cur_lr_list.append(cur_lr)
    for epoch in range(max_epoch):
        for i,batch in enumerate(range(iters)):
            optimizer.step()
            # scheduler.step(epoch + batch / iters)
        cur_lr = optimizer.param_groups[-1]['lr']
        cur_lr_list.append(cur_lr)
        lr_scheduler.step_update(epoch * iters + i)
        print('epoch: {},cur_lr: {}'.format(epoch,cur_lr))
    x_list = list(range(len(cur_lr_list)))
    plt.title('Cosine lr  T_0:{}  T_mult:{}'.format(T_0,T_mult))
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.plot(x_list, cur_lr_list)
    plt.savefig('./lr.png')
import torch.optim
from config import *
import  timm.scheduler as TScheduler
if __name__=='__main__':
    model=models.resnet18(pretrained=False)
    T_0=3
    T_mult=2
    args = args_resnet50_afterPreTrain
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,betas=(0.9,0.999),eps=1e-08, amsgrad=False)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    #scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    n_iter_per_epoch = 70000//256
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
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-6, last_epoch=-1)
   #  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6,
   #                                                                   last_epoch=-1)
    visulize_cosine_lr(model,400,optimizer,scheduler,n_iter_per_epoch)