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

train_size =224


def plot_embedding_2D(data, label, title,batch_size):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    # fig = plt.figure()

    fig = plt.figure()

    plt.axis('off')
    colors = ['LightPink','NavajoWhite','Purple','MediumPurple','Gray','ForestGreen','CornflowerBlue','DarkTurquoise','MediumAquamarine','Sienna','Crimson','Moccasin','DarkSlateGray','Orange']
    # colors = ['#614E52', '#614E52', '#614E52', '#614E52', '#614E52', '#FF0000', '#FF0000',
    #           '#FF0000', '#FF0000', '#FF0000', '#FF0000', 'Moccasin', 'DarkSlateGray', 'Orange']

    # 正常
    # colors  = ["#614E52"] * n + ["#A68E76"] * n + ["#FF0000"] * n + ["#A68E76"] * n + ["#B17Aaa"] * n + [
    #     "#2D381F"] * n + ["#E2CDBC"] * n + ["#92ACD1"] * n + ["#DAA520"] * n + ["#A52A2A"] * n
    # 恶意
    # colors = ["#614E52""#FF0000"] * n + ["#A68E76"] * n + ["#A68E76"] * n + ["#B17Aaa"] * n + [
    #     "#2D381F"] * n + ["#E2CDBC"] * n + ["#92ACD1"] * n + ["#DAA520"] * n + ["#A52A2A"] * n

    for i in range(data.shape[0]):
        # num = int(i / 100)
        # plt.scatter(data[i, 0], data[i, 1], color=str(clor[i]), marker='o', alpha=1, s=2)

        # print(num)
        plt.text(data[i, 0], data[i, 1], str("●"),# str(int(label[i])), # 用什么符号绘制当前数据点
                 color= colors[label[i]], # plt.cm.Set1(int(label[i]) / 10),
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

from data_helper import MyDataset
from config import *


if __name__ == '__main__':
    test_data_dir = '../data/new_split_data/val'
    test_datasets = MyDataset(test_data_dir, 'val',input_size=384,resize=400)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=128, num_workers=4)

    args = args_convnext_tiny
    arch = args['name']

    args['model_hyperparameters']['num_classes'] = 128
    model = load_model(args['name'], args['model_hyperparameters'])
    dim_mlp = model.num_features
    model.head.fc = torch.nn.Sequential(
        torch.nn.Linear(dim_mlp,dim_mlp),
        torch.nn.ReLU(),
        model.head.fc
    )
    model.load_state_dict(torch.load('../model_data/convnext_t_ssl_simclr/Convnext_T_epoch48_acc91.36324541284404_valAcc76.921875.pth',map_location='cpu'))
    model.eval()
    model = model.cuda()
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)  # 调用TSNE
    features = None
    labels = None
    with torch.no_grad():
        for i, (inputs, label) in enumerate(tqdm(test_dataloader)):
            inputs = inputs.cuda()
            feature = model(inputs)
            feature= feature.detach().clone().cpu()
            if features is None:
                labels = label
                features = feature.detach().clone().cpu()
            else:
                features = torch.cat([features,feature])
                labels = torch.cat([labels,label])
        result_2D = tsne_2D.fit_transform(features)
        fig1 = plot_embedding_2D(result_2D, labels, 'bac_Model', 16)
        # os.makedirs('../model_data/convnext_t_ssl_simclr/TSNE',exist_ok=True)
        fig1.savefig('../model_data/convnext_t_ssl_simclr/tsne.png')
    # main()

