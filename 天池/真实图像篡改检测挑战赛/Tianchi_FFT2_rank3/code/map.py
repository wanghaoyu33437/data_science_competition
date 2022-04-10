import os
import cv2
from tqdm import tqdm
import shutil
import timm
import timm.data.transforms
import numpy as np
# test1_mask_path = '../save_out/test/senet154/0.9263_AllImg_S3_500_adamw_BCE07_Dice03/images_merge_768_896_score_2655/'
test1_mask_path = '../save_out/test/senet154/Mar_13_1.2956_AllImg_decompose768_s3_data_500_newDataAug_senet154/images_merge_896_score2752/'
# test2_mask_path = '../save_out/best_2743/images/'
test2_mask_path = '../save_out/test/senet154/Mar_09_1.2995_AllImg_decompose768_val500_s3_data500_adamw_CAWR_bec07_dice03_focal1_pathsize768/images_merge_896_score2757/'
# test2_mask_path = '../save_out/best_2655/'
test3_mask_path =  '../save_out/test/senet154/Mar_18_train1.7438_val1.1071_valLoss0.2411_AllImg_s3data_2k_bookscreen_1446_size768_newDataAug_senet154_bce07_dice03_focal1/images_score2756/'
test4_mask_path =  '../save_out/test/senet154/Mar_15_train1.6746_val1.1101_valLoss0.2485_AllImg_s3_data_newDataAug_senet154_dice03_bce07_focal1/images_merge_896_score2766/'
test5_mask_path = '../save_out/test/senet154/Mar_16_train1.6967_val1.0601_valLoss0.2611_AllImg_s3_data_1k_bookscreen_579_size896_newDataAug_senet154_dice03_bce07_focal1/images_merge_896_score2764/'



test1_paths = sorted(os.listdir(test1_mask_path))
test2_paths =sorted(os.listdir(test2_mask_path))
# test3_paths =sorted(os.listdir(test3_mask_path))

bad_example_current = np.load('current_bad.npy')
# bad_example_baseline = np.load('senet154_0.8870_com_merge1_bad.npy')

path_name ='Mar_18_2756_AND_Mar_16_2764_And_Mar_15_2766_AND_Mar_13_2752_Mar_9_2757_th2000/'

def OR(path=path_name):
    save_path = '../save_out/map/OR/'+path+'/images/'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    for test in tqdm(test1_paths):
        mask1 = cv2.imread(test1_mask_path+test)
        mask2 = cv2.imread(test2_mask_path+test)
        # mask3 = cv2.imread(test3_mask_path+test)
        cv2.imwrite(save_path+test,(mask1|mask2))
def XOR():
    save_path = '../save_out/map/XOR/images/'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    for test in tqdm(test1_paths):
        mask1 = cv2.imread(test1_mask_path+test)
        mask2 = cv2.imread(test2_mask_path+test)
        # mask3 = cv2.imread(test3_mask_path+test)
        cv2.imwrite(save_path+test,(mask1^mask2))
def AND(path=path_name):
    a,b,c= 0,0,0
    d= 0
    save_path = '../save_out/map/AND/'+path+'/images/'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    for test in tqdm(test1_paths):
        mask1 = cv2.imread(test1_mask_path + test)
        mask2 = cv2.imread(test2_mask_path + test)
        mask3 = cv2.imread(test3_mask_path + test)
        mask4 = cv2.imread(test4_mask_path + test)
        mask5 = cv2.imread(test5_mask_path + test)
        mask1[mask1>=100]=255
        mask1[mask1<100]=0
        mask2[mask2>=100]=255
        mask2[mask2<100]=0
        mask3[mask3>=100]=255
        mask3[mask3<100]=0
        mask4[mask4 >= 100] = 255
        mask4[mask4 < 100] = 0
        mask5[mask5 >= 100] = 255
        mask5[mask5 < 100] = 0
        # m12 = mask1&mask2
        # m13 = mask1&mask3
        # m23 = mask2&mask3
        mask = mask1/255+mask2/255+mask4/255+mask5/255+mask3/255
        # mask = mask2/255+mask4/255+mask5/255
        mask_or = mask.copy()
        mask_or[mask_or >= 1] = 255
        mask_or[mask_or < 1] = 0
        mask[mask >= 3] = 255 # 投票>=2 选中
        mask[mask < 3] = 0

        # mask3 = cv2.imread(test3_mask_path+test)
        if (mask==255).sum()<=2000:
            cv2.imwrite(save_path+test,mask_or.astype(np.uint8))
            a+=1
        else:
            b+=1
            cv2.imwrite(save_path + test, mask.astype(np.uint8))
    print('And:%d,Or:%d'%(b,a))
        # if ((mask1&mask2)==255).sum()<6000 :
        #     cv2.imwrite(save_path+test,(mask1|mask2))
        #     a+=1
        # else:
        #     cv2.imwrite(save_path + test, (mask1&mask2))

AND()

def copy():
    a=0
    b=0
    for test in tqdm(test1_paths):
        name = test[:-4]+'.png'
        if name in bad_example_current:
            mask1 = cv2.imread(test1_mask_path+test)
            mask2 = cv2.imread(test2_mask_path+ test)
            if (mask1==255).sum() < (mask2==255).sum():
                shutil.copy(test2_mask_path + test, test1_mask_path + test)
                b+=1
    print(a,b)