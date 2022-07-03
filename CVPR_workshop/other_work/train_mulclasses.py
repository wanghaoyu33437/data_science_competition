# from __future__ import print_function
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# import random
# import shutil
# from tqdm import tqdm
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.utils.data as data
# from transform import gen_transform
# import torchvision
# import torchvision.datasets as datasets
# from config import *
# from utils import AverageMeter,load_model, accuracy,rm_and_make_dir
# import pandas as pd
# from models.resnet import *
# from imageAugement import ImageCorrupter
# seed = 7777
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# import torch.cuda
# torch.backends.cudnn.deterministic = True
#
# import logging
# # pre_train
# # train_size = 368
# train_size = 224
#
# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self,train_file,mode='train',normal_num=0,num_classes=2):
#         self.num= len(train_file)
#         self.train_file = train_file
#         # self.labels = label_file
#         self.mode = mode
#         self.normal_num=normal_num
#         self.num_classes = num_classes
#         self.transform = gen_transform(resize=train_size, mode=mode)
#         self.ImageCorrupter =ImageCorrupter(prob=0.2,n=1)
#         logging.info(self.transform)
#     def __getitem__(self, index):
#         image,label= self.load_item(index)
#         return image, label
#     def __len__(self):
#         return self.num
#     def mixup(self,image,label):
#         mixup_idx = random.randint(0, self.num-1)
#         image_path, mixup_target = self.train_file[mixup_idx]
#         mixup_label = np.zeros([self.num_classes]).astype(np.float32)
#         mixup_label[int(mixup_target)] = 1
#         mixup_image = cv2.imread(image_path)[:, :, ::-1]
#         # mixup_image = cv2.resize(mixup_image,(train_size,train_size))
#         # Select a random number from the given beta distribution
#         # Mixup the images accordingly
#         transformed = self.transform(image=mixup_image)
#         mixup_image = transformed['image']
#         alpha = 0.2
#         lam = np.random.beta(alpha, alpha)
#         image = lam * image + (1 - lam) * mixup_image
#         label = lam * label + (1 - lam) * mixup_label
#         return image, label
#     def load_item(self,index):
#         if self.mode != 'val':
#             image_path,target = self.train_file[index]
#             if not os.path.exists(image_path):
#                 print(image_path)
#             image = cv2.imread(image_path)[:,:,::-1]
#             image = cv2.resize(image,(train_size,train_size))
#             label = np.zeros([self.num_classes]).astype(np.float32)
#             label[int(target)]=1
#             if image_path.endswith('JPEG'):
#                 image = self.ImageCorrupter(image)
#             transformed = self.transform(image=image)
#             image = transformed['image']
#             '''mixup'''
#             if random.random()<train_args.MIXUP:
#                 image,label= self.mixup(image,label)
#                 label = label.astype(np.float32)
#         else:
#             image_path, target = self.train_file[index]
#             image = cv2.imread(image_path)[:, :, ::-1]
#             label = np.zeros([self.num_classes]).astype(np.float32)
#             label[int(target)] = 1
#             transformed = self.transform(image=image)
#             image = transformed['image']
#         return image,label
#
# def cross_entropy(outputs, smooth_labels):
#     loss = torch.nn.KLDivLoss(reduction='batchmean')
#     return loss(F.log_softmax(outputs, dim=1), smooth_labels)
#
# def labelSmoothing(target,smoothing,classes=20,dim=-1):
#     confidence = 1.0 - smoothing
#     true_dist=torch.zeros_like(torch.tensor(np.random.random((target.shape[0], classes))))
#     true_dist.fill_(smoothing / (classes - 1))
#     true_dist.scatter_(1, target.detach().cpu().unsqueeze(1), confidence)
#     return true_dist
# def main():
#     arch = args['name']
#     if args['batch_size'] > 256:
#         # force the batch_size to 256, and scaling the lr
#         args['optimizer_hyperparameters']['lr'] *= 256 / args['batch_size']
#         args['batch_size'] = 256
#     # Data
#     print('Loading dataset....')
#     # adv =np.load('./Phase1_Traindata/MixImage/Adv_30_15_Pat_corruption68961.npy',allow_pickle=True)[22987:]
#     # clean= np.load('./Phase1_Traindata/MixImage/AllTrain_corruption_50k_x224.npy',allow_pickle=True)
#     # train_file = np.concatenate([clean,adv[22987:22987]])
#     # adv_08 = np.load('./Phase1_Traindata/ADVImage/train_adv008_path_22987.npy', allow_pickle=True)
#     # train_file.extend(list(np.load('./Phase1_Traindata/ADVImage/train_adv_path_22987.npy', allow_pickle=True)))
#     # train_file.extend(list(np.load('./Phase1_Traindata/ADVImage/train_adv030_MTI_path_22987.npy',allow_pickle=True)))
#
#     # train_file = np.load('./Phase1_Traindata/CorruptionImage/clean_imagecorruption_68k.npy',allow_pickle=True)
#     # train_file = np.concatenate([train_file,np.load('./Phase1_Traindata/CorruptionImage/adv008_015_020_patch_imagecorruption.npy',allow_pickle=True)[::2]])
#
#     train_file = np.load('./Phase2_traindata/CleanImage/clean.npy', allow_pickle=True)[:22000:] #0
#     val_file = np.load('./Phase2_traindata/CleanImage/clean.npy', allow_pickle=True)[22000::2]
#
#     train_file = np.concatenate(
#         [train_file, np.load('./Phase2_traindata/ADVImage/adv4.npy', allow_pickle=True)[:22000:]]) # 1
#     val_file = np.concatenate([val_file, np.load('./Phase2_traindata/ADVImage/adv4.npy', allow_pickle=True)[22000::2]])
#
#     train_file = np.concatenate([train_file, np.load('./Phase2_traindata/ADVImage/adv4_l2_2.npy', allow_pickle=True)[:22000:]])# 2
#
#     val_file = np.concatenate([val_file, np.load('./Phase2_traindata/ADVImage/adv4_l2_2.npy', allow_pickle=True)[22000::2]])
#
#     train_file = np.concatenate([train_file, np.load('./Phase2_traindata/ADVImage/adv_deepfool_2.npy', allow_pickle=True)[:11000]])  # 11800张
#     val_file = np.concatenate([val_file, np.load('./Phase2_traindata/ADVImage/adv_deepfool_2.npy', allow_pickle=True)[11000::2]])
#
#     train_file = np.concatenate([train_file, np.load('./Phase2_traindata/ADVImage/advPatch_3.npy', allow_pickle=True)[:22000:]]) #3
#     val_file = np.concatenate([val_file, np.load('./Phase2_traindata/ADVImage/advPatch_3.npy', allow_pickle=True)[22000::2]])
#
#     train_file = np.concatenate([train_file, np.load('./Phase2_traindata/ADVImage/adv_corruption_4.npy', allow_pickle=True)[:22000:]])  # 4
#     val_file = np.concatenate( [val_file, np.load('./Phase2_traindata/ADVImage/adv_corruption_4.npy', allow_pickle=True)[22000::2]])
#     print('Loading dataset successfully')
#     print("TrainData num: %d "%len(train_file))
#     TrainDataSet = MyDataset(train_file=train_file,mode='mix',normal_num=len(train_file),num_classes=5)
#     trainloader = data.DataLoader(TrainDataSet, batch_size=args['batch_size'], shuffle=True, num_workers=4)
#
#
#     # val_file = np.load('./Phase2_traindata/adv183.npy', allow_pickle=True)
#     valDataSet =  MyDataset(train_file=val_file,mode='val',normal_num=len(val_file),num_classes=5)
#     valloader = data.DataLoader(valDataSet,batch_size=32,shuffle=False,num_workers=4)
#
#     # Model
#     model = load_model(arch,num_classes=5)
#     # in_features = model.module.fc.in_features
#     # out_features = model.module.fc.out_features
#     # model.module.fc= nn.Sequential(
#     #         nn.Linear(in_features,2048),
#     #         nn.BatchNorm1d(2048),
#     #         nn.ReLU(),
#     #         nn.Linear(2048, out_features)
#     #     )
#     # model = model.half()
#     if args['resume'] ==True:
#         print('Load preTrain weight')
#         model.load_state_dict(torch.load(args['resume_path']))
#
#     logger.info('Trainging :' + arch)
#     logger.info(args)
#     # print('Trainging :' + arch)
#     best_acc = 0  # best test accuracy
#
#     optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
#                                                        **args['optimizer_hyperparameters'])
#     if args['scheduler_name'] != None:
#         scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
#                                                                               **args['scheduler_hyperparameters'])
#     model = model.cuda()
#     # Train and val
#     train_accs = []
#     val_accs = []
#     train_losses = []
#     val_losses = []
#     for epoch in tqdm(range(args['epochs'])):
#         train_loss, train_acc = train(trainloader, model, optimizer,scheduler,epoch)
#         train_accs.append(train_acc)
#         train_losses.append(train_loss)
#         val_acc,val_loss = val(valloader,model)
#         val_accs.append(val_acc)
#         val_losses.append(val_loss)
#         # print('Train acc: {}; Test acc{}'.format(train_acc,test_acc))
#         logging.info('Epoch: %d, train acc:%f, train loss:%f' % (epoch+1,train_acc, train_loss))
#         logging.info( 'Epoch: %d, val acc:%f, val loss：%f' % (epoch+1,val_acc,val_loss))
#         # logging.info('Test acc:%f, Test loss:%f' % (test_acc, test_loss))
#         # save model
#         best_acc = max(train_acc, best_acc)
#         if train_acc>=60:
#             save_checkpoint(model=model,arch=arch+'_epoch%d_T%5.4f_V%5.4f'%(epoch+1,train_acc,val_acc))
#         if args['scheduler_name'] != None:
#             scheduler.step()
#     train_history = pd.DataFrame({
#         'acc': train_accs,
#         'loss': train_losses
#     })
#     train_history.to_csv(os.path.join(current_path, arch + '_train.csv'))
#     logging.info("Best acc:{}".format(best_acc))
#
# def val(valloader, model):
#     losses = AverageMeter()
#     accs = AverageMeter()
#     model.eval()
#     with torch.no_grad():
#         for (inputs, labels) in tqdm(valloader):
#             inputs, labels = inputs.cuda(), labels.cuda()
#             targets = labels.argmax(dim=1)
#             outputs = model(inputs)
#             loss = cross_entropy(outputs, labels)
#             acc = accuracy(outputs, targets)
#             losses.update(loss.item(), inputs.size(0))
#             accs.update(acc[0].item(), inputs.size(0))
#     torch.cuda.empty_cache()
#     return accs.avg,losses.avg
#
#
# def train(trainloader, model, optimizer,scheduler,epoch):
#     losses = AverageMeter()
#     accs = AverageMeter()
#     model.eval()
#     num_steps= len(trainloader)
#     # switch to train mode
#     model.train()
#     for i,(inputs, labels) in enumerate(tqdm(trainloader)):
#         # break
#         inputs = inputs.to(torch.float32).cuda()# .half()
#         labels = labels.cuda()
#         targets = labels.argmax(dim=1)
#         if train_args.LS != 0:
#             labels= labelSmoothing(targets,train_args.LS,classes=2)
#             labels = labels.to(torch.float32).cuda()
#         # soft_labels = labels.cuda()
#         optimizer.zero_grad()
#
#         outputs = model(inputs)
#         # print(outputs)
#         # print(outputs.argmax(1))
#         loss = cross_entropy(outputs, labels)
#         acc = accuracy(outputs, targets)
#
#         loss.backward()
#         optimizer.step()
#         losses.update(loss.item(), inputs.size(0))
#         accs.update(acc[0].item(), inputs.size(0))
#     return losses.avg, accs.avg
#
#
# def save_checkpoint(model, arch):
#     filepath = os.path.join(current_path, arch + '.pth')
#     torch.save(model.state_dict(), filepath)
#
# import time
# import argparse
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--MIXUP',type=float,default=0.2,help='prob of using mixup')
#     parser.add_argument('--LS', type=float, default=0, help='prob of using label Smoothing')
#     parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
#     train_args= parser.parse_args()
#     args = args_resnet50
#     # args = args_convnext_base#args_swin_tiny_patch4_window7_224#,args_swin_s3_tiny_224,args_convnext_tiny_hnf
#     TIME = time.asctime().split(' ')
#     date = TIME[1]+'_'+TIME[2]+'_'
#     params = 'Resnet50_mulClassesTrain_clean_22k_adv4_22k_adv4_l2_22k_deepfool_11k_AdvCor_22k_Patch_22k_size%d_epochs'%(train_size)
#     current_path = "./Phase2_model_data/"+date+params
#     rm_and_make_dir(current_path)
#     logger = logging.getLogger(__name__)
#     logging.basicConfig(
#         format='[%(asctime)s] - %(message)s',
#         datefmt='%Y/%m/%d %H:%M:%S',
#         level=logging.DEBUG,
#         handlers=[
#             logging.FileHandler(os.path.join(current_path, 'train.log')),
#             logging.StreamHandler()
#         ])
#     logging.info(current_path)
#     logging.info('在resnet50后加mlp,数据组成：clean 22k,0 corruption_prob0.2; adv4 22k 1; adv4 l2 22k deepfool 11k 2; patch 22k 3; adv corruptiong 22k 4')
#     # if not os.path.exists(os.path.join(current_path,'config1.py')):
#     shutil.copy('config.py', os.path.join(current_path, 'config.py'))
#     shutil.copy('train_mulclasses.py', os.path.join(current_path, 'train_mulclasses.py'))
#     main()
