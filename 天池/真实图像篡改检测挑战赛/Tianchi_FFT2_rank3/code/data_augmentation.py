import random
import numpy as np
import math
import cv2
def rand_bbox(size):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat_w = random.random() * 0.3 + 0.05
    cut_rat_h = random.random() * 0.3 + 0.05

    cut_w = int(W * cut_rat_w)
    cut_h = int(H * cut_rat_h)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
"""
Online
"""
def copy_move(img, img2, msk):
    '''
    :param img: 待修改图像，tensor :[3,H,W] H=W
    :param img2: 如果是None，就从自身copy一块，若不是就从img2 copy，[3,H,W]
    :param msk: [1,H,W]
    :return:
    '''
    #  resize = A.Resize(512, 512)(image=img, mask=msk)
    #     # img = torch.from_numpy(resize['image']).permute(2, 0, 1)
    #     # msk = torch.from_numpy(resize['mask'])
    size = img.size()
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    if img2 is None:
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.size())
        x_move = random.randrange(-bbx1, (W - bbx2))
        y_move = random.randrange(-bby1, (H - bby2))
        img[:, bbx1 + x_move:bbx2 + x_move, bby1 + y_move:bby2 + y_move] = img[:, bbx1:bbx2, bby1:bby2]
        msk[:,bbx1 + x_move:bbx2 + x_move, bby1 + y_move:bby2 + y_move] = 1.0
        # img = cv2.rectangle(img.numpy().transpose(1, 2, 0), pt1=[bby1 + y_move, bbx1 + x_move],
        #                     pt2=[bby2 + y_move, bbx2 + x_move], color=(255, 0, 0), thickness=5)
    else:
        # resize = A.Resize(512, 512)(image=img2)
        # img2 = torch.from_numpy(resize['image']).permute(2, 0, 1)
        assert img.shape == img2.shape
        bbx1, bby1, bbx2, bby2 = rand_bbox(img2.size())
        x_move = random.randrange(-bbx1, (W - bbx2))
        y_move = random.randrange(-bby1, (H - bby2))
        img[:, bbx1 + x_move:bbx2 + x_move, bby1 + y_move:bby2 + y_move] = img2[:, bbx1:bbx2, bby1:bby2]
        msk[:,bbx1 + x_move:bbx2 + x_move, bby1 + y_move:bby2 + y_move] = 1.0
        # img = cv2.rectangle(img.numpy().transpose(1, 2, 0), pt1=[bby1 + y_move, bbx1 + x_move],
        #                     pt2=[bby2 + y_move, bbx2 + x_move], color=(255, 0, 0), thickness=3)
    return img, msk
"""
offline
"""
def copy_move_offline(img, msk=None):
    H, W, C = img.shape
    if msk == None:
        msk = np.zeros_like(img, dtype="uint8")
    bbx1, bby1, bbx2, bby2 = rand_bbox([C,H,W])
    x_move = random.randrange(-bbx1, (W - bbx2))
    y_move = random.randrange(-bby1, (H - bby2))
    img[bby1 + y_move:bby2 + y_move, bbx1 + x_move:bbx2 + x_move,:] = img[bby1:bby2,bbx1:bbx2,:]
    msk[bby1 + y_move:bby2 + y_move, bbx1 + x_move:bbx2 + x_move,:] = 255
    # img = cv2.rectangle(img.copy(), pt1=[bby1 + y_move, bbx1 + x_move])
    return img,msk
        #
def random_paint(image,mask=None):
    mask_ = np.zeros(image.shape[:2], dtype="uint8")
    # x0, y0 = (700, 50)
    # x1, y1 = (0, 50)
    # x2, y2 = (0, 120)
    # x3, y3 = (700, 120)

    x0, y0 = (700, 200)
    x1, y1 = (0, 200)
    x2, y2 = (0, 300)
    x3, y3 = (700, 300)
    if mask==None:
        mask = np.zeros_like(image, dtype="uint8")
    mask[y1:y2, x1:x0,:] = 255
    x_mid0, y_mid0 = int((x1 + x2) / 2), int((y1 + y2) / 2)
    x_mid1, y_mi1 = int((x0 + x3) / 2), int((y0 + y3) / 2)
    thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    cv2.line(mask_, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
    img = cv2.inpaint(image, mask_, 7, cv2.INPAINT_NS)
    # img = cv2.rectangle(img, pt1=[x2, y1], pt2=[x3, y2], color=(255, 0, 0), thickness=3)
    return img,mask
import os
from utils import rm_and_make_dir
from tqdm import tqdm

def augementatio(path_out_img,path_out_mask,img_tmp,idx):
    img, msk = copy_move_offline(img_tmp)
    cv2.imwrite(path_out_img + '%d.jpg' % idx, img)
    cv2.imwrite(path_out_mask + '%d.png' % idx, msk)
    idx += 1
    img, msk = random_paint(img_tmp.copy())
    cv2.imwrite(path_out_img + '%d.jpg' % idx, img)
    cv2.imwrite(path_out_mask + '%d.png' % idx, msk)
    idx += 1
    return idx

def decompose(data_path='/home2/WHY/04_FakeImageDetection/Tianchi_FFT2_rank3/data/book_screen/screen/'):
    path = data_path
    flist = sorted(os.listdir(path))
    size_list = [768]
    # path_out = '/home2/WHY/04_FakeImageDetection/Tianchi_FFT2_rank3/data/train_896/mask_%d/'
    path_out_img = '/home2/WHY/04_FakeImageDetection/Tianchi_FFT2_rank3/data/book_screen/book_sceen_train/'
    path_out_mask = '/home2/WHY/04_FakeImageDetection/Tianchi_FFT2_rank3/data/book_screen/book_sceen_mask/'
    # rm_and_make_dir(path_out_img)
    # rm_and_make_dir(path_out_mask)
    rtn_list = [[], [], [], []]
    idx =646
    # idx = len(os.listdir('/home2/WHY/04_FakeImageDetection/Tianchi_FFT2_rank3/data/book_screen/book_sceen_train/'))
    for file in tqdm(flist):
        img = cv2.imread(path + file)
        H, W, _ = img.shape
        size_idx= 0
        while size_idx <len(size_list)-1:
            if H<size_list[size_idx+1] or W < size_list[size_idx+1]:
                break
            size_idx += 1
        rtn_list[size_idx].append(file)
        size = size_list[size_idx]
        X, Y = H // size + 1, W // size + 1
        for x in range(X-1):
            for y in range(Y-1):
                img_tmp = img[x * size: (x + 1) * size, y * size: (y + 1) * size, :]
                idx=augementatio(path_out_img,path_out_mask,img_tmp.copy(),idx)

            img_tmp = img[x * size: (x + 1) * size, -size:, :]
            idx = augementatio(path_out_img, path_out_mask, img_tmp.copy(), idx)
            # cv2.imwrite(path_out_img + '%d.png' % idx, img_tmp)
            # idx += 1
        for y in range(Y - 1):
            img_tmp = img[-size:, y * size: (y + 1) * size, :]
            # cv2.imwrite(path_out_img + '%d.png' % idx, img_tmp)
            idx = augementatio(path_out_img, path_out_mask, img_tmp.copy(), idx)
            # idx += 1
        img_tmp = img[-size:, -size:, :]
        # cv2.imwrite(path_out_img + '%d.png' % idx, img_tmp)
        idx = augementatio(path_out_img, path_out_mask,img_tmp.copy(),idx)
        # idx += 1
