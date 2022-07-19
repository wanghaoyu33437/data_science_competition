import glob
import logging
import shutil

import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets import ImageFolder,VisionDataset
from torchvision.datasets.folder import  default_loader,make_dataset,IMG_EXTENSIONS
import cv2
import random
import os
from tqdm import tqdm
import numpy as np
from transforms import gen_transform
from PIL import Image
TRAIN = 'train'
VAL = 'val'
TEST = 'test'
class MyDataset(VisionDataset):
    def __init__(self,root: str,mode='train',input_size=256,resize=224,extensions=IMG_EXTENSIONS,
                 is_valid_file= None) -> None:
        super(MyDataset, self).__init__(root)
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS if is_valid_file is None else None, is_valid_file)
        self.transform =gen_transform(input_size,resize,mode)
        self.mode = mode
        logging.info(self.transform)
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx,
        extensions = None,
        is_valid_file= None,
    ):
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def _find_classes(self, dir: str):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = cv2.imread(path)[::,::,::-1]
        if 'ssl' in self.mode:
            if self.transform is not None:
                sample = [self.transform(image=sample)['image'] for _ in range(2)]
        else:
            if self.transform is not None:
                transformed = self.transform(image=sample)
                sample = transformed['image']
                # sample= self.transform(sample)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

if __name__ == '__main__':
    train_data_dir = './data/new_data/train'
    val_data_dir = './data/new_data/val'
    # dataset = MyDataset(train_data_dir,'train')
    # dataset.transform =None
    # val_data_dir = './data/new_data/val'
    # val_dataset = MyDataset(val_data_dir,'train')
    # val_dataset.transform =None
    classes = os.listdir(train_data_dir)
    for c in classes:
        imgs = os.listdir(os.path.join(train_data_dir,c),)
        imgs.sort()
        os.makedirs(os.path.join(val_data_dir, c),exist_ok=True)
        for img in imgs:
            if random.random()<=0.1:
                shutil.move(os.path.join(train_data_dir,c,img),os.path.join(val_data_dir,c,img))
                # shutil.move(os.path.join(train_data_dir, c, img.replace('_1','_2')), os.path.join(val_data_dir, c, img.replace('_1','_2')))




    # classes = os.listdir(path)
    # num = 400
    # for c in classes:
    #     imgs = glob.glob(os.path.join(path,c,'*'))
    #     prob = num/len(imgs)
    #     for img in imgs:
    #         if random.random() <=prob:
    #             shutil.move(img,img.replace('train','val'))

    # train_path = './data/new_split4_data/train'
    # # val_path= './data/new_split4_data/val'
    # # os.makedirs(val_path,exist_ok=True)
    # os.makedirs(train_path, exist_ok=True)
    # classes = dataset.classes
    # # dataset =train_dataset+val_dataset
    # for i,(x,y) in enumerate(tqdm(dataset)):
    #     H, W, C = x.shape
    #     x1, x2 = x[:int(H/2+0.5), :int(W / 2 + 0.5), ::-1], x[:int(H/2+0.5), int(W / 2 + 0.5):, ::-1],
    #     x3, x4 = x[int(H/2+0.5):, :int(W / 2 + 0.5), ::-1], x[int(H/2+0.5):, int(W / 2 + 0.5):, ::-1],
    #
    #     # if random.random()<=0.1:
    #     #     class_dir = os.path.join(val_path,classes[y])
    #     #     os.makedirs(class_dir, exist_ok=True)
    #     #     cv2.imwrite(os.path.join(class_dir,str(i)+'_1.png'),x1)
    #     #     cv2.imwrite(os.path.join(class_dir, str(i) + '_2.png'), x2)
    #     # else :
    #     class_dir = os.path.join(train_path,classes[y])
    #     os.makedirs(class_dir, exist_ok=True)
    #     cv2.imwrite(os.path.join(class_dir,str(i)+'_1.png'),x1)
    #     cv2.imwrite(os.path.join(class_dir, str(i) + '_2.png'), x2)
    #     cv2.imwrite(os.path.join(class_dir,str(i)+'_3.png'),x3)
    #     cv2.imwrite(os.path.join(class_dir, str(i) + '_4.png'), x4)

    # dataloader  = torch.utils.data.DataLoader(train_dataset,batch_size=4,shuffle=True)