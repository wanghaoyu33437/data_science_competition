import numpy as np
import os

data_train_768 = np.load('/home2/WHY/04_FakeImageDetection/project/user_data/flist/flist1/train_768.npy')
data_val_768 = np.load('/home2/WHY/04_FakeImageDetection/project/user_data/flist/flist1/val_768.npy')

data_s3_data = np.concatenate([np.load('/home2/WHY/04_FakeImageDetection/project/user_data/flist/flist_add/train_s3data_1800.npy'),
                               np.load('/home2/WHY/04_FakeImageDetection/project/user_data/flist/flist_add/val_s3data_200.npy')
                               ])


