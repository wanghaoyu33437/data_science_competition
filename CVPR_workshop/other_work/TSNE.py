from models.resnet_torch import  *
from torch import nn
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import random
import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import load_model
from sklearn.manifold import TSNE
from transform import *
train_size =224

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
        self.transform = gen_transform(resize=train_size, mode=mode)
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
        anchor_label = np.zeros([2]).astype(np.float32)
        anchor_label[int(target)]=1

        positive_idx = random.randint(0, self.positive_num - 1)
        positive_path,positive_target = self.positive_file[positive_idx]
        positive_label = np.zeros([2]).astype(np.float32)
        positive = cv2.imread(positive_path)[:,:,::-1]
        positive = cv2.resize(positive,(train_size,train_size))
        positive_label[int(positive_target)] = 1

        negative_idx = random.randint(0, self.negative_num - 1)
        negative_path, negative_target = self.negative_file[negative_idx]
        negative_label = np.zeros([2]).astype(np.float32)
        negative = cv2.imread(negative_path)[:, :, ::-1]
        negative = cv2.resize(negative, (train_size, train_size))
        negative_label[int(negative_target)] = 1
        ''' 
        以adv样本为anchor
        '''
        # if self.anchor=='adv':
        #     negative = self.ImageCorrupter(negative)
        # else:
        #     anchor = self.ImageCorrupter(anchor)
        #     positive = self.ImageCorrupter(positive)
        transformed = self.transform(image=anchor)
        anchor = transformed['image']
        transformed = self.transform(image=positive)
        positive = transformed['image']
        transformed = self.transform(image=negative)
        negative = transformed['image']
        return anchor,positive,negative,anchor_label,positive_label,negative_label
'''
database 十个类正常图片
test 十个类正常图片 + 一个类的后门图片
train十个类的正常图片 + 每个类的后门图片
# color : https://blog.csdn.net/daichanglin/article/details/1563299
'''


def plot_embedding_2D(data, label, title,batch_size):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    # fig = plt.figure()

    fig = plt.figure()

    plt.axis('off')
    # colors = ['LightPink','NavajoWhite','Purple','MediumPurple','Gray','ForestGreen','CornflowerBlue','DarkTurquoise','MediumAquamarine','Sienna','Crimson','Moccasin','DarkSlateGray','Orange']
    # colors = ['#614E52', '#614E52', '#614E52', '#614E52', '#614E52', '#FF0000', '#FF0000',
    #           '#FF0000', '#FF0000', '#FF0000', '#FF0000', 'Moccasin', 'DarkSlateGray', 'Orange']
    n = batch_size
    # 正常
    # colors  = ["#614E52"] * n + ["#A68E76"] * n + ["#FF0000"] * n + ["#A68E76"] * n + ["#B17Aaa"] * n + [
    #     "#2D381F"] * n + ["#E2CDBC"] * n + ["#92ACD1"] * n + ["#DAA520"] * n + ["#A52A2A"] * n
    # 恶意
    colors = ["#614E52"] * n + ["#FF0000"] * n + ["#A68E76"] * n + ["#A68E76"] * n + ["#B17Aaa"] * n + [
        "#2D381F"] * n + ["#E2CDBC"] * n + ["#92ACD1"] * n + ["#DAA520"] * n + ["#A52A2A"] * n

    for i in range(data.shape[0]):
        # num = int(i / 100)
        # plt.scatter(data[i, 0], data[i, 1], color=str(clor[i]), marker='o', alpha=1, s=2)

        # print(num)
        plt.text(data[i, 0], data[i, 1], str("●"),# str(int(label[i])), # 用什么符号绘制当前数据点
                 color=  "#FF0000" if label[i] == 1 else"#614E52", # plt.cm.Set1(int(label[i]) / 10),
                 fontdict={'weight': 'bold', 'size': 9})
        # plt.scatter(data[i, 0], data[i, 1], color=str(colors[i]), marker='o', alpha=1, s=2)
    plt.show()
    # fig.savefig('tsne.png')
    # plt.xticks([])
    # plt.yticks([])
    # plt.title(title)
    return fig

def plot_embedding_3D(data,label,title):
    # 将数据进行归一化处理
    x_min, x_max = np.min(data,axis=0), np.max(data,axis=0)
    data = (data- x_min) / (x_max - x_min)
    plt.axis('off')
    ax = plt.figure().add_subplot(111,projection='3d')

    # colors = ['LightPink', 'NavajoWhite', 'Purple', 'MediumPurple', 'Gray', 'ForestGreen', 'CornflowerBlue',
    #           'DarkTurquoise', 'MediumAquamarine', 'Sienna', 'Crimson', 'Moccasin', 'DarkSlateGray', 'Orange']
    colors = ['#614E52', '#A56C41', '#191970', '#A68E76', '#B17Aaa', '#2D381F', 'E2CDBC',
              '#92ACD1', '#DAA520', '#A52A2A', '#FF0000', 'Moccasin', 'DarkSlateGray', 'Orange']
    for i in range(data.shape[0]):
        num = int(i / 130)
        ax.text(data[i, 0], data[i, 1], data[i,2],str("●"), color=colors[num],fontdict={'weight': 'bold', 'size': 9})
    # for i in range(data.shape[0]):
    #     plt.scatter(data[i, 0], data[i, 1], color=str(colors[i]), marker='o', alpha=1, s=2)
    # plt.show()
    return ax
def plot_embedding(data, clean_n, poison_n, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    plt.axis('off')
    if title == 'clean':
        n = clean_n
        clor = ["#614E52"] * n + ["#A56C41"] * n + ["#191970"] * n + ["#A68E76"] * n + ["#B17Aaa"] * n + [
            "#2D381F"] * n + ["#E2CDBC"] * n + ["#92ACD1"] * n + ["#DAA520"] * n + ["#A52A2A"] * n
    else:
        assert title == 'poison'
        n = clean_n
        clor = ["#614E52"] * n + ["#A56C41"] * n + ["#191970"] * n + ["#A68E76"] * n + ["#B17Aaa"] * n + [
            "#2D381F"] * n + ["#E2CDBC"] * n + ["#92ACD1"] * n + ["#DAA520"] * n + ["#A52A2A"] * n + [
                   '#FF0000'] * poison_n

    # print(title)
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1], color=str(clor[i]), marker='o', alpha=1, s=2)

    plt.show()




if __name__ == '__main__':
    # model = resnet50(pretrained=False,num_classes = 2)
    # model = nn.DataParallel(model).cuda()
    model = load_model('resnet50',num_classes=2)
    model.module.fc = nn.Sequential()
    '''swin transform'''
    model_path = 'Apr_19_Resnet50_Triple_Train_clean_all_adv4_Patch_advl2_advCor_size224/'
    ckpt_path = 'Phase2_model_data/' + model_path + 'resnet50_epoch%d_TF1_00_VF1_00_TP0_FP0.pth'
    print(ckpt_path)


    clean_file = np.load('./Phase2_traindata/CleanImage/clean.npy',allow_pickle=True)
    adv_file1 = np.load('./Phase2_traindata/ADVImage/adv4.npy', allow_pickle=True)
    adv_file2 = np.load('./Phase2_traindata/ADVImage/advPatch.npy', allow_pickle=True)
    adv_file3 = np.load('./Phase2_traindata/ADVImage/adv4_l2.npy', allow_pickle=True)
    adv_file4 = np.load('./Phase2_traindata/ADVImage/adv_corruption.npy', allow_pickle=True)
    TrainDataSet1 = TripleDataset(positive_file=adv_file2,
                                  negative_file=clean_file
                                  , anchor='clean', mode='val',
                                 normal_num=22987)
    trainloader = torch.utils.data.DataLoader(TrainDataSet1, batch_size=128, shuffle=True, num_workers=4)
    for i, (anchor, positive, negative, a,b,c) in enumerate(tqdm(trainloader)):
        # break
        batch_size = len(anchor)
        anchor = anchor.to(torch.float32).cuda()  # .half()
        positive = positive.to(torch.float32).cuda()
        negative = negative.to(torch.float32).cuda()
        inputs = torch.cat([anchor, positive, negative])
        labels = torch.cat([a,b,c])
        # break
        break
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)  # 调用TSNE
    for epoch in range(1,95):
        model.load_state_dict(torch.load(ckpt_path%epoch, map_location='cpu'))
        model = model.eval()
        model=model.cuda()
        with torch.no_grad():
            feature = model(inputs)
            feature = feature.detach().clone().cpu()
        #     anchor_f, positive_f, negative_f = feature[:batch_size], feature[batch_size:batch_size * 2], feature[batch_size * 2:]
        #
        result_2D = tsne_2D.fit_transform(feature)
        #     tsne_3D = TSNE(n_components=3, init='pca', random_state=0)
        #     result_3D = tsne_3D.fit_transform(feature)
        #     print('Finished......')
        #     # 调用上面的两个函数进行可视化
        fig1 = plot_embedding_2D(result_2D, labels.argmax(1), 'bac_Model', 16)
        os.makedirs('user_data/TSNE_advPatch',exist_ok=True)
        fig1.savefig('user_data/TSNE_advPatch/%d.png'%epoch)
    # main()


