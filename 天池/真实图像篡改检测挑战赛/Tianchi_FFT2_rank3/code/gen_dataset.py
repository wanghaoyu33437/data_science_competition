import numpy as np
import os
from sklearn.model_selection import KFold,train_test_split
from PIL import Image
from utils import rm_and_make_dir
from data_augmentation import copy_move_offline,random_paint
import cv2
import numpy as np
import random

import shutil
from torchvision import transforms
from tqdm import tqdm

import torch
import torch.cuda


def decompose(data_path='/home2/WHY/04_FakeImageDetection/Tianchi_FFT2_rank3/data/book_screen/book/'):
    path = data_path
    flist = sorted(os.listdir(path))
    size_list = [768]
    # path_out = '/home2/WHY/04_FakeImageDetection/Tianchi_FFT2_rank3/data/train_896/mask_%d/'
    path_out = '/home2/WHY/04_FakeImageDetection/Tianchi_FFT2_rank3/data/book_screen/book_%d/'
    for size in size_list:
        rm_and_make_dir(path_out%size)
    rtn_list = [[], [], [], []]
    for file in tqdm(flist):
        img = cv2.imread(path + file)
        H, W, _ = img.shape
        size_idx= 0
        while size_idx <len(size_list)-1:
            if H<size_list[size_idx+1] or W < size_list[size_idx+1]:
                break
            size_idx += 1
        # size_idx = 0
        # while size_idx < len(size_list) - 1:
        #     if H < size_list[size_idx+1] and  W < size_list[size_idx+1]:
        #         break
        #     size_idx += 1
        rtn_list[size_idx].append(file)
        # size = size_list[size_idx]
        size = size_list[size_idx]
        # print(file, H, W, size)
        # if H>=size_list[size_idx+1] and W>= size_list[size_idx+1]:
        #     print(file,H, W,size)
        # path_out = '../data/inputs/'
        X, Y = H // size + 1, W // size + 1
        idx = 0
        for x in range(X-1):
            for y in range(Y-1):
                img_tmp = img[x * size: (x + 1) * size, y * size: (y + 1) * size, :]
                img,msk = copy_move_offline
                cv2.imwrite(path_out%size + file[:-4] + '_%03d.png' % idx, img_tmp)
                idx += 1
            img_tmp = img[x * size: (x + 1) * size, -size:, :]
            cv2.imwrite(path_out%size + file[:-4] + '_%03d.png' % idx, img_tmp)
            idx += 1
        for y in range(Y - 1):
            img_tmp = img[-size:, y * size: (y + 1) * size, :]
            cv2.imwrite(path_out%size + file[:-4] + '_%03d.png' % idx, img_tmp)
            idx += 1
        img_tmp = img[-size:, -size:, :]
        cv2.imwrite(path_out%size + file[:-4] + '_%03d.png' % idx, img_tmp)
        idx += 1
    return rtn_list
# def merge(path_d,path_r,size_list):
#     test_path = '/home2/WHY/04_FakeImageDetection/baseline/data/test/img/'
#     # path_d = '../save_out/test/senet154/Mar_9_1.2995_AllImg_decompose768_val500_s3_data500_adamw_CAWR_bec07_dice03_focal1_pathsize768/images_compose_tta_HF_OR_896/'
#     # path_r = '../save_out/test/senet154/Mar_9_1.2995_AllImg_decompose768_val500_s3_data500_adamw_CAWR_bec07_dice03_focal1_pathsize768/images_merge_tta_HF_OR_896/'
#     rm_and_make_dir(path_r)
#     # size_list = size_list
#     val_lists =os.listdir(test_path)
#     bad_examples = []
#     for file in tqdm(val_lists):
#         img = np.array(Image.open(test_path+file))
#         # img = cv2.imread(test_path + file)
#         H, W, _ = img.shape
#         size_idx =0
#         while size_idx <len(size_list)-1:
#             if H<size_list[size_idx+1] or W < size_list[size_idx+1]:
#                 break
#             size_idx += 1
#         size = size_list[size_idx]
#         X, Y = H // size + 1, W // size + 1
#         idx = 0
#         rtn = np.zeros((H, W, 3), dtype=np.uint8)
#         for x in range(X-1):
#             for y in range(Y-1):
#                 # img_tmp = cv2.imread(path_d + file[:-4] + '_%03d.png' % idx)
#                 img_tmp = np.array(Image.open(path_d + file[:-4] + '_%03d.png' % idx))
#                 rtn[x * size: (x + 1) * size, y * size: (y + 1) * size, :] += img_tmp
#                 idx += 1
#             # img_tmp = cv2.imread(path_d + file[:-4] + '_%03d.png' % idx)
#             img_tmp = np.array(Image.open(path_d + file[:-4] + '_%03d.png' % idx))
#             rtn[x * size: (x + 1) * size, -size:, :] += img_tmp
#             idx += 1
#         for y in range(Y - 1):
#             # img_tmp = cv2.imread(path_d + file[:-4] + '_%03d.png' % idx)
#             img_tmp = np.array(Image.open(path_d + file[:-4] + '_%03d.png' % idx))
#             rtn[-size:, y * size: (y + 1) * size, :] += img_tmp
#             idx += 1
#         # img_tmp = cv2.imread(path_d + file[:-4] + '_%03d.png' % idx)
#         img_tmp = np.array(Image.open(path_d + file[:-4] + '_%03d.png' % idx))
#
#         rtn[-size:, -size:, :] += img_tmp
#         idx += 1
#         rtn[rtn>=200]=255
#         if (rtn == 255).sum()<=12000:
#             bad_examples.append(file[:-4] + '.png')
#         Image.fromarray(rtn).save(path_r + file[:-4] + '.png')
#         # cv2.imwrite(path_r + file[:-4] + '.png', rtn)
#     # np.save('./Mar_9_1.2995_bad.npy', bad_examples)
#     print("Len of bad is %d "% len(bad_examples))

def merge(path,size_list):
    test_path = '/home2/WHY/04_FakeImageDetection/baseline/data/test/img/'
    # rm_and_make_dir(path)
    val_lists =os.listdir(test_path)
    for file in tqdm(val_lists):
        img = cv2.imread(test_path + file)
        H, W, _ = img.shape
        size_idx =0
        while size_idx <len(size_list)-1:
            if H<size_list[size_idx+1] or W < size_list[size_idx+1]:
                break
            size_idx += 1
        size = size_list[size_idx]
        X, Y = H // size + 1, W // size + 1
        idx = 0
        rtn = np.zeros((H, W, 3), dtype=np.uint8)
        for x in range(X-1):
            for y in range(Y-1):
                img_tmp = cv2.imread(path + file[:-4] + '_%03d.png' % idx)
                os.remove(path + file[:-4] + '_%03d.png' % idx)
                rtn[x * size: (x + 1) * size, y * size: (y + 1) * size, :] += img_tmp
                idx += 1
            img_tmp = cv2.imread(path + file[:-4] + '_%03d.png' % idx)
            os.remove(path + file[:-4] + '_%03d.png' % idx)
            rtn[x * size: (x + 1) * size, -size:, :] += img_tmp
            idx += 1
        for y in range(Y - 1):
            img_tmp = cv2.imread(path + file[:-4] + '_%03d.png' % idx)
            os.remove(path + file[:-4] + '_%03d.png' % idx)
            rtn[-size:, y * size: (y + 1) * size, :] += img_tmp
            idx += 1
        img_tmp = cv2.imread(path + file[:-4] + '_%03d.png' % idx)
        os.remove(path + file[:-4] + '_%03d.png' % idx)
        rtn[-size:, -size:, :] += img_tmp
        idx += 1
        rtn[rtn>=200]=255
        cv2.imwrite(path + file[:-4] + '.png', rtn)
import glob
def gen_fold(train_path):
    kf1 = KFold(n_splits=4,shuffle=True)
    train_path = '/home2/WHY/04_FakeImageDetection/Tianchi_FFT2_rank3/data/'
    train_base_path = '/home2/WHY/04_FakeImageDetection/baseline/data/train/img'
    inputs_paths = os.listdir(train_base_path)
    # inputs_decompose_paths = os.listdir(train_path+'inputs')
    # inputs=[]
    # for path in inputs_decompose_paths:
    #     inputs.extend(os.listdir(os.path.join(train_path+'inputs',path)))
    # inputs.sort()
    inputs_paths = np.array(inputs_paths)
    # masks_decompose_paths = os.listdir(train_path+'masks')
    # masks=[]
    # for path in masks_decompose_paths:
    #     masks.extend(os.listdir(os.path.join(train_path+'masks',path)))
    # masks.sort()
    # masks = np.array(masks)
    if not os.path.exists('../user_data'):
        os.mkdir('../user_data')
    for i,(train_idx,val_idx) in enumerate(kf1.split(inputs_paths,inputs_paths)):
        train_inputs = inputs_paths[train_idx]
        val_inputs = inputs_paths[val_idx]
        # train_mask = masks[train_idx]
        # val_inputs = inputs[val_idx]
        # val_mask= masks[val_idx]
        train_lists= []
        for j in tqdm(range(len(train_inputs))):
            inputs_decompose_paths =glob.glob(train_path+'inputs/' +train_inputs[j][:-4]+'_*')
            masks_decompose_paths = glob.glob(train_path+'masks/' +train_inputs[j][:-4]+'_*')
            train_lists.extend([a,b] for a,b in zip(inputs_decompose_paths,masks_decompose_paths))
        val_lists = []
        for j in tqdm(range(len(val_inputs))):
            inputs_decompose_paths = glob.glob(train_path + 'inputs/' + val_inputs[j][:-4] + '_*')
            masks_decompose_paths = glob.glob(train_path + 'masks/' + val_inputs[j][:-4] + '_*')
            val_lists.extend([a, b] for a, b in zip(inputs_decompose_paths, masks_decompose_paths))
        if not os.path.exists('../user_data/flist1'):
            os.mkdir('../user_data/flist1')
        np.save('../user_data/flist1/fold_%d_train.npy'%i,train_lists)
        np.save('../user_data/flist1/fold_%d_val.npy' %i, val_lists)
def gen_all_train_and_val():
    train_path = '../data/train_768/'
    mask_path = '../data/mask_768/'
    train_paths = sorted(os.listdir(train_path))
    mask_paths = sorted(os.listdir(mask_path))
    train_list = []
    val_list = []
    val_index = np.random.choice(np.arange(len(train_paths)),size=int(len(train_paths)/10),replace=False)
    for i,(img_name, mask_name) in tqdm(enumerate(zip(train_paths[:], mask_paths[:]))): #
        mask = cv2.imread(mask_path+mask_name,cv2.IMREAD_GRAYSCALE)
        if (mask==255).sum()<=5000:
            continue
        if i in val_index:
            val_list.append([train_path + img_name, mask_path + mask_name])
        else:
            train_list.append([train_path + img_name, mask_path + mask_name])

    if not os.path.exists('../user_data/flist1'):
        os.mkdir('../user_data/flist1')
    print('train size is %d,val size is %d'%(len(train_list),len(val_list)))
    np.save('../user_data/flist1/train_768_noALLBlack.npy', train_list)
    np.save('../user_data/flist1/val_768_noALLBlack.npy', val_list)
# rtn_list=decompose()
def gen_addition():
    train_path = '../data/book_screen/book_sceen_train/'
    mask_path = '../data/book_screen/book_sceen_mask/'
    train_paths = sorted(os.listdir(train_path))
    mask_paths = sorted(os.listdir(mask_path))
    train_list = []
    val_list = []
    # val_index = np.random.choice(np.arange(100),size=20,replace=False)
    for i,(img_name, mask_name) in tqdm(enumerate(zip(train_paths[:], mask_paths[:]))):  #
        # if i in val_index:
        #     val_list.append([train_path + img_name, mask_path + mask_name])
        # else:
        if random.random()<=0.4:
            train_list.append([train_path + img_name, mask_path + mask_name])

    if not os.path.exists('../user_data/flist_add'):
        os.mkdir('../user_data/flist_add')
    print('train size is %d,val size is %d'%(len(train_list),len(val_list)))
    np.save('../user_data/flist_add/train_book_screen_%d.npy'%len(train_list), train_list)
    # np.save('../user_data/flist_add/val_erasing_10.npy', val_list)




def gen_new_map(fold_name=''):
    fold_path ='../user_data/flist1/'
    data = np.load(fold_path+fold_name)
    new_data = []
    for input, mask in data:
        input = input.replace('/home2/WHY/04_FakeImageDetection/Tianchi_FFT2_rank3', '..')
        mask = mask.replace('/home2/WHY/04_FakeImageDetection/Tianchi_FFT2_rank3', '..')
        new_data.append([input,mask])
    np.save('../user_data/flist1/'+fold_name,new_data)

# fold_paths = os.listdir('/home/dell/HARD-DATA/WHY/04_FakeImageDetection/Tianchi_FFT2_rank3/user_data/flist1')
# for f in fold_paths:
#     gen_new_map(f)

def put_mask(img_path,mask_path,output_fold,name):
    # 1.读取图片
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    H, W, _ = image.shape
    start_x = random.randint(0,W-201)
    start_y = random.randint(0,H-201)
    if random.random()>=0.5:
        width = random.randint(100,W-start_x-1)
        height = random.randint(100,200)
    else:
        height = random.randint(100,H-start_y-1)
        width =random.randint(100,200)
    bbox1 = [start_x,start_y,start_x+width,start_y+height]
    color = [(0,0,0),(255,0,0),(255,255,255),(0,0,255)]
    mask[start_y:start_y+height,start_x:start_x+width,:] = 255
    # 3.画出mask
    zeros = np.zeros((image.shape), dtype=np.uint8)
    zeros_mask = cv2.rectangle(zeros, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                    color=color[random.randint(0,3)], thickness=-1 ) #thickness=-1 表示矩形框内颜色填充
    zeros_mask = np.array(zeros_mask)
    try:
    	# alpha 为第一张图片的透明度
        alpha = 1
        # beta 为第二张图片的透明度
        beta = 1
        gamma = 0
        # cv2.addWeighted 将原始图片与 mask 融合
        if random.random()>=0.5:
            mask_img = cv2.addWeighted(image, alpha, zeros_mask, beta, 0)
        else:
            image[bbox1[1]:bbox1[3],bbox1[0]:bbox1[2],:]=color[random.randint(0,3)]
            mask_img =image
        os.makedirs(os.path.join(output_fold, 'mask'), exist_ok=True)
        os.makedirs(os.path.join(output_fold, 'img'), exist_ok=True)
        cv2.imwrite(os.path.join(output_fold,'img',name), mask_img)
        cv2.imwrite(os.path.join(output_fold, 'mask',name), mask)
    except:
        print('异常')
    # return mask_img

def add_new_train():
    train_path = '../../baseline/data/train/img/'
    mask_path = '../../baseline/data/train/mask/'
    train_paths = sorted(os.listdir(train_path))
    output_fold='../../baseline/data/add_Erasing/'
    if os.path.exists(output_fold):
        shutil.rmtree(output_fold)
    os.mkdir(output_fold)
    # transform = transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(, 0, 0))
    mask_paths = sorted(os.listdir(mask_path))
    index = np.random.choice(np.arange(4000), size=100, replace=False)
    print(index)
    for i in tqdm(index):
        print(mask_paths[i])
        put_mask(train_path+train_paths[i],mask_path+mask_paths[i],output_fold,mask_paths[i])
# add_new_train()

def caclulate_dataset():
    train_path = '../../baseline/data/train/mask/'
    mask_names= os.listdir(train_path)
    true_list=[]
    false_list=[]
    for name in tqdm(mask_names):
        mg = Image.open(train_path+name)
        mg_np= np.array(mg)
        true_list.append((mg_np==255).sum())
        false_list.append((mg_np==0).sum())
    return np.mean(true_list),np.mean(false_list)






