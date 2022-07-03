import pandas as pd
from utils.utils import  *
from tqdm import tqdm
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']= '3'
import torch.utils.data
from transform import gen_transform
import cv2
test_size = 368
class Mydataset(torch.utils.data.Dataset):
    def __init__(self,test_file,test_path):
        self.test_file = test_file
        self.test_path = test_path
        self.transform = gen_transform(input_size=test_size,mode='val')
        pass
    def __getitem__(self, index):
        imageA_name = val_csv['imageA'][index]
        imageB_name = val_csv['imageB'][index]
        imageA_path = os.path.join(self.test_path,val_csv['imageA'][index])
        imageB_path = os.path.join(self.test_path,val_csv['imageB'][index])
        imageA = cv2.imread(imageA_path)[:,:,::-1]
        imageB = cv2.imread(imageB_path)[:, :, ::-1]
        transformed = self.transform(image=imageA)
        imageA = transformed['image']
        transformed = self.transform(image=imageB)
        imageB = transformed['image']
        return imageA,imageB,imageA_name,imageB_name
        pass
    def __len__(self):
        return len(self.test_file)
val_csv = pd.read_csv('dataset/pet_biometric_challenge_2022/validation/new_valid_data.csv')
result  = pd.read_csv('dataset/pet_biometric_challenge_2022/validation/valid_data.csv')
path = 'dataset/pet_biometric_challenge_2022/validation/images/'
# convnext_tiny_hnf
args = dict(num_classes=6000,drop_path_rate=0.0)
''' 测试阶段，只要backbone部分'''
model = timm.create_model('convnext_tiny_hnf',**args)
# model = load_model('resnet50',num_classes=6000)
#
model_path = 'May_10_Convnext_t_train96k_val24k_size368_Mixup0_LS0/'
ckpt_path = 'Model_data/'+model_path+'convnext_tiny_hnf_epoch48_T95.5965_V93.0165.pth'
print(ckpt_path)
ckpt= torch.load(ckpt_path,map_location='cpu')

new_ckpt = dict()
for (key1, value1), (key2, value2) in zip(ckpt.items(), model.named_parameters()):
    new_ckpt[key2] = value1

model.load_state_dict(new_ckpt)
model = model.cuda()
model.eval()


testDataset = Mydataset(val_csv,path)
dataLoader = torch.utils.data.DataLoader(testDataset,batch_size=16,num_workers=4)
index  = 0
with torch.no_grad():
    for imgA,imgB,imgA_names,imgB_names in tqdm(dataLoader):
        bs = imgB.shape[0]
        imgA = imgA.cuda()
        imgB = imgB.cuda()
        features_A = model.forward_features(imgA)
        features_B = model.forward_features(imgB)
        logit_A = model.forward_head(features_A,True)
        logit_B = model.forward_head(features_B,True)
        confidence = cos_simi(logit_A,logit_B)
        confidence = confidence.detach().cpu().numpy()
        for c in confidence:
            result.loc[index,'prediction']= c
            index+=1
# #
output = './output_data/'+model_path+ckpt_path.split('/')[-1][:-4]+'/'
os.makedirs(output,exist_ok=True)
result.to_csv(output+'result_cosine%d.csv'%test_size,index=False)
# result.to_csv(output+'result_Euclidean.csv',index=False)