import pandas as pd
import shutil
import os
from tqdm import tqdm
import glob
import numpy as np
import random
from sklearn.model_selection import train_test_split
image_paths =  '../dataset/pet_biometric_challenge_2022/class_image/train_add/'
category = os.listdir(image_paths)
category.sort()
train_datapath_category = []
val_datapath_category = []
for c in category:
    images=os.listdir(image_paths+c)
    target = int(c)
    for img in images:
        if random.random()<=0.2:
            val_datapath_category.append([os.path.join(image_paths[3:],c,img),target])
        else:
            train_datapath_category.append([os.path.join(image_paths[3:],c,img),target])
os.makedirs('../user_data/train_data_add/',exist_ok=True)

np.save('../user_data/train_data_add/train_add_96k.npy',train_datapath_category)


# preds = pd.read_csv('../DataSet\pet_biometric_challenge_2022/validation/valid_data.csv')
# for i in tqdm(range(len(preds))):
#     preds.loc[i, 'imageA'] = preds.loc[i, 'imageA'].replace('*', '_')
#     name = preds['imageA'][i]
#     if not  os.path.exists('../DataSet\pet_biometric_challenge_2022/validation/images/'+name):
#         print(name)
#     preds.loc[i, 'imageB'] = preds.loc[i, 'imageB'].replace('*', '_')
#     name = preds['imageB'][i]
#     if not  os.path.exists('../DataSet\pet_biometric_challenge_2022/validation/images/'+name):
#         print(name)
#
#     # path ='../DataSet/pet_biometric_challenge_2022/class_image/train/%d/'%ID
#     # os.makedirs(path,exist_ok=True)
#     #
#     # shutil.copy('../DataSet/pet_biometric_challenge_2022/train/images/'+name,path+name)
#
# preds.to_csv('../DataSet\pet_biometric_challenge_2022/validation/new_valid_data.csv')
