import albumentations as A
import os
import cv2
from tqdm import tqdm
import shutil
import random
resize = 256

transform =A.Compose([
            A.SomeOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.RandomContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.GridDistortion(p=0.5),
                A.Transpose(p=0.5),
                A.CoarseDropout(30,15,15,p=0.5),
                A.ElasticTransform(0.5),
        ],n=2)])

image_paths =  './dataset/pet_biometric_challenge_2022/class_image/train/'
out_dir = './dataset/pet_biometric_challenge_2022/class_image/train_add/'
os.makedirs(out_dir,exist_ok=True)
category = os.listdir(image_paths)
category.sort()
datapath_category = []
num_of_category =20
for c in tqdm(category[:]):
    image_names=os.listdir(image_paths+c)
    target = int(c)

    os.makedirs(os.path.join(out_dir,c),exist_ok=True)
    for i,name in enumerate(image_names):
        shutil.copy(os.path.join(image_paths,c,name),os.path.join(out_dir,c,'%d_%d.png')%(target,i))
    current_category_img_num = i+1
    while current_category_img_num<num_of_category:
        for name in image_names:
            img = cv2.imread(os.path.join(image_paths,c,name))
            transformed =transform(image = img)
            img = transformed['image']
            cv2.imwrite(os.path.join(out_dir,c,'%d_%d.png'%(target,current_category_img_num)),img)
            current_category_img_num+=1
            if current_category_img_num==num_of_category:
                break




