from __future__ import print_function
import os
# from train import MyDataset
os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
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
from utils.utils import AverageMeter,load_model, accuracy,rm_and_make_dir
import pandas as pd
from TSNE import *



seed = 7777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
import torch.cuda
torch.backends.cudnn.deterministic = True
import logging
# pre_train
# train_size = 368
train_size = 368

scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast

class TripleDataset(torch.utils.data.Dataset):
    def __init__(self,positive_file,negative_file,mode='mix',anchor='normal',normal_num=0):
        self.positive_num= len(positive_file)
        self.negative_num= len(negative_file)
        self.positive_file = positive_file
        self.negative_file = negative_file
        # self.labels = label_file
        self.mode = mode
        self.anchor = anchor
        self.normal_num=normal_num
        self.transform = gen_transform(resize = 400,input_size=368, mode=mode)
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
        anchor_label = np.array([int(target)]).astype(np.float32)

        positive_idx = random.randint(0, self.positive_num - 1)
        positive_path,positive_target = self.positive_file[positive_idx]

        positive = cv2.imread(positive_path)[:,:,::-1]
        positive = cv2.resize(positive,(train_size,train_size))
        positive_label =  np.array([int(positive_target)]).astype(np.float32)

        negative_idx = random.randint(0, self.negative_num - 1)
        negative_path, negative_target = self.negative_file[negative_idx]
        negative = cv2.imread(negative_path)[:, :, ::-1]
        negative = cv2.resize(negative, (train_size, train_size))
        negative_label =  np.array([int(negative_target)]).astype(np.float32)

        transformed = self.transform(image=anchor)
        anchor = transformed['image']
        transformed = self.transform(image=positive)
        positive = transformed['image']
        transformed = self.transform(image=negative)
        negative = transformed['image']
        return anchor,positive,negative,anchor_label,positive_label,negative_label

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
    trainloader_list = []
    valloader_list= []
    clean_file = np.load('./Phase2_traindata/CleanImage/clean.npy',allow_pickle=True)
    adv_file1 = np.load('./Phase2_traindata/ADVImage/adv4.npy', allow_pickle=True)
    # adv_file1 = np.concatenate([np.load('./Phase2_traindata/ADVImage/adv4.npy', allow_pickle=True)[::2],
    #                             np.load('./Phase2_traindata/ADVImage/adv4_perturbation.npy', allow_pickle=True)[1::2]])
    adv_file2 = np.load('./Phase2_traindata/ADVImage/advPatch.npy', allow_pickle=True)
    adv_file3 =np.load('./Phase2_traindata/ADVImage/adv4_l2.npy', allow_pickle=True)
    adv_file4 = np.load('./Phase2_traindata/ADVImage/adv_corruption.npy', allow_pickle=True)
    # adv_file3 = np.concatenate([np.load('./Phase2_traindata/ADVImage/adv4_l2.npy', allow_pickle=True),
    #                             np.load('./Phase2_traindata/ADVImage/adv4_l2_perturbation.npy', allow_pickle=True)[1::2]])
    TrainDataSet1 = TripleDataset(positive_file=adv_file1[::],
                                  negative_file=clean_file[::], anchor='clean', mode='mix',
                                 normal_num=22987)
    trainloader_list.append(data.DataLoader(TrainDataSet1, batch_size=args['batch_size'], shuffle=True, num_workers=4))
    TrainDataSet2 = TripleDataset(positive_file=adv_file2[::],
                                  negative_file=np.concatenate([clean_file[::]])
                                  , anchor='adv1', mode='mix',
                                  normal_num=22987)
    trainloader_list.append(data.DataLoader(TrainDataSet2, batch_size=args['batch_size'], shuffle=True, num_workers=4))

    TrainDataSet3 = TripleDataset(positive_file=adv_file3[:],
                                  negative_file=np.concatenate([clean_file[:]])
                                  , anchor='adv2', mode='mix',
                                  normal_num=22987)
    trainloader_list.append(data.DataLoader(TrainDataSet3, batch_size=args['batch_size'], shuffle=True, num_workers=4))

    TrainDataSet4 = TripleDataset(positive_file=adv_file4[::],
                                  negative_file=np.concatenate([clean_file[::]])
                                  , anchor='adv2', mode='mix',
                                  normal_num=22987)
    trainloader_list.append(data.DataLoader(TrainDataSet4, batch_size=args['batch_size'], shuffle=True, num_workers=4))

    val_file = np.load('Phase2_traindata/val_4k.npy',allow_pickle=True)
    valDataSet =  MyDataset(train_file=val_file,mode='val',normal_num=len(val_file))
    valloader = data.DataLoader(valDataSet,batch_size=32,shuffle=False,num_workers=4)

    print('Loading dataset successfully')

    # Model
    model = load_model(arch,num_classes=2)

    # model = resnet50(pretrained=False,num_classes = 1)
    # model = model.half()
    if args['resume'] ==True:
        print('Load preTrain weight')
        model.load_state_dict(torch.load(args['resume_path']))
    model.module.fc = nn.Sequential()
    logger.info('Trainging :' + arch)
    logger.info(args)
    # print('Trainging :' + arch)
    best_acc = 0  # best test accuracy
    optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
                                                       **args['optimizer_hyperparameters'])
    if args['scheduler_name'] != None:
        scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
                                                                              **args['scheduler_hyperparameters'])
    # model = nn.DataParallel(model).cuda()
    model = model.cuda()
    # Train and val
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(args['epochs'])):
        if epoch == 0:
            val(valloader,model,epoch)
        train_loss_list,p,r,train_F1 = train(trainloader_list, model, optimizer,scheduler,epoch)
        # train_accs.append(train_acc)
        # train_losses.append(train_loss)
        with torch.no_grad():
            F1, TP, FP = val(valloader, model,epoch+1)
        # val_accs.append(val_acc)
        # val_losses.append(val_loss)
        # print('Train acc: {}; Test acc{}'.format(train_acc,test_acc))
        logging.info('Epoch: %d, train_triple_loss:%f,train_cls_loss:%f' % (epoch+1,np.mean(train_loss_list,0)[0],np.mean(train_loss_list,0)[1]))
        logging.info('Epoch: %d, train F1:%f, train Precision:%f,train Recall:%f' % (epoch+1,train_F1, p,r))
        logging.info('Epoch: %d, val F1 :%f, TP:%d,FP:%d' % (epoch + 1, F1, TP, FP))
        # logging.info('Test acc:%f, Test loss:%f' % (test_acc, test_loss))
        # save model
        # best_acc = max(train_acc, best_acc)
        # if train_acc>=75:
        save_checkpoint(model=model, arch=arch + '_epoch%d_TF1_%02.f_VF1_%02.f_TP%d_FP%d' % (epoch + 1, train_F1 * 100, F1 * 100, TP, FP))
        if args['scheduler_name'] != None:
            scheduler.step()
    train_history = pd.DataFrame({
        'acc': train_accs,
        'loss': train_losses
    })
    train_history.to_csv(os.path.join(current_path, arch + '_train.csv'))
    logging.info("Best acc:{}".format(best_acc))

def train(trainloader_list, model, optimizer,scheduler,epoch):
    loss_clss = AverageMeter()
    loss_tris = AverageMeter()
    accs = AverageMeter()
    model.eval()
    num_steps= len(trainloader_list[0])
    # switch to train mode
    model.train()
    TP,FP,TN,FN = 0,0,0,0
    p, r, F1 = 0,0,0

    loss_list=[]
    loss_cls,loss_tri = 0 ,0

    # triplet_loss =  nn.TripletMarginWithDistanceLoss(distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y))
    triplet_loss = nn.TripletMarginLoss(margin=3)
    for index_l,l in enumerate(trainloader_list):
        loss_tris = AverageMeter()
        for i,(anchor,positive,negative,label1,label2,label3) in enumerate(tqdm(l)):
            batch_size = len(anchor)
            anchor = anchor.to(torch.float32).cuda()# .half()
            positive= positive.to(torch.float32).cuda()
            negative = negative.to(torch.float32).cuda()
            inputs = torch.cat([anchor,positive,negative])
            labels = torch.cat([label1,label2,label3])
            # labels  =labels.view(-1,labels.shape[-1])/
            labels = labels.cuda()
            # targets = labels.argmax(dim=1)
            optimizer.zero_grad()
            with autocast():
                # outputs,feature = model(inputs)
                feature = model(inputs)
                anchor_f,positive_f,negative_f= feature[:batch_size],feature[batch_size:batch_size*2],feature[batch_size*2:]
                loss_tri = triplet_loss(anchor=anchor_f,positive=positive_f,negative=negative_f)
                # loss_cls = F.binary_cross_entropy_with_logits(outputs, labels)
            # predict = F.sigmoid(outputs)
            # TP += ((predict >= 0.5) & (labels == 1)).cpu().numpy().sum()
            # FP += ((predict >= 0.5) & (labels == 0)).cpu().numpy().sum()
            # TN += ((predict <= 0.5) & (labels == 0)).cpu().numpy().sum()
            # FN += ((predict <= 0.5) & (labels == 1)).cpu().numpy().sum()
            # acc = accuracy(outputs, targets)
            loss = loss_tri
            print("\nAdv_file%d  tri_loss%f,cls_loss%f\n"%(index_l+1,loss_tri,loss_cls))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            loss_tris.update(loss_tri.item(), inputs.size(0))
            # logging.info(loss_tri.item())
            # loss_clss.update(loss_cls.item(), inputs.size(0))
            # accs.update(0, inputs.size(0))
        loss_list.append([loss_tris.avg,loss_clss.avg])
    # p = TP / (TP + FP)
    # r = TP / (TP + FN)
    # F1 = 2 * r * p / (r + p)
        # logging.info('Dataloder_%d,triplet_loss:%f'%(i+1,loss_tris.avg))
    return loss_list,p,r,F1

def val(valloader, model,epoch):
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)  # 调用TSNE
    model.eval()
    TP,FP,TN,FN = 0,0,0,0
    p, r, F1 = 0,0,0
    feature_all= None
    labels_all =None
    with torch.no_grad():
        for (inputs, labels) in tqdm(valloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            targets = labels.argmax(dim=1)
            # outputs,feature = model(inputs)
            feature = model(inputs)
            if feature_all is None:
                labels_all = labels.detach().clone().cpu()
                feature_all = feature.detach().clone().cpu()
            else:
                labels_all = torch.cat([labels_all, labels.detach().clone().cpu()])
                feature_all= torch.cat([feature_all,feature.detach().clone().cpu()])
            # predict = F.sigmoid(outputs)
            # TP += ((predict >= 0.5) & (labels == 1)).cpu().numpy().sum()
            # FP += ((predict >= 0.5) & (labels == 0)).cpu().numpy().sum()
            # TN += ((predict <= 0.5) & (labels == 0)).cpu().numpy().sum()
            # FN += ((predict <= 0.5) & (labels == 1)).cpu().numpy().sum()
    # p = TP / (TP + FP)
    # r = TP / (TP + FN)
    # F1 = 2 * r * p / (r + p)
    result_2D = tsne_2D.fit_transform(feature_all)
    print(labels_all)
    fig = plot_embedding_2D(result_2D,labels_all, '',64)
    fig.savefig(os.path.join(current_path, 'tsne_epoch_%d.png'%epoch))
    return F1,TP,FP

def save_checkpoint(model, arch):
    filepath = os.path.join(current_path, arch + '.pth')
    torch.save(model.state_dict(), filepath)

import time
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
    parser.add_argument('--MIXUP',type=float,default=0.2,help='prob of using mixup')
    parser.add_argument('--LS', type=float, default=0, help='prob of using label Smoothing')
    train_args= parser.parse_args()
    args = args_resnet50_afterPreTrain
    # args = args_convnext_base#args_swin_tiny_patch4_window7_224#,args_swin_s3_tiny_224,args_convnext_tiny_hnf
    TIME = time.asctime().split(' ')
    date = TIME[1]+'_'+TIME[2]+'_'
    params = 'Resnet50_Triple_Train_clean_all_adv4_Patch_advl2_advCor_size%d'%(train_size)
    current_path = "./Phase2_model_data/"+date+params
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
    logging.info('数据组成：clean,all,corruption_prob0.2,adv4，patch_1/2')
    # if not os.path.exists(os.path.join(current_path,'config1.py')):
    shutil.copy('config.py', os.path.join(current_path, 'config.py'))
    shutil.copy('train_triple.py', os.path.join(current_path, 'train_triple.py'))
    main()
