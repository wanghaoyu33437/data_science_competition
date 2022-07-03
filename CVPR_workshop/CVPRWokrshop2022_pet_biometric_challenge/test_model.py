import pandas as pd
from utils.utils import  *
from tqdm import tqdm
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']= '0,1'
import torch.utils.data
from transform import gen_transform
from transform import gen_transform
import cv2
train_size = 224
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,train_file,mode='train',normal_num=0,num_classes=2):
        self.num= len(train_file)
        self.train_file = train_file
        # self.labels = label_file
        self.mode = mode
        self.normal_num=normal_num
        self.num_classes = num_classes
        self.transform = gen_transform(resize=train_size, mode=mode)

    def __getitem__(self, index):
        image,label= self.load_item(index)
        return image, label
    def __len__(self):
        return self.num
    def load_item(self,index):
        image_path, target = self.train_file[index]
        image = cv2.imread(image_path)[:, :, ::-1]
        label = np.zeros([self.num_classes]).astype(np.float32)
        label[int(target)] = 1
        transformed = self.transform(image=image)
        image = transformed['image']
        return image,label

# convnext_tiny_hnf
args = dict(num_classes=6000,drop_path_rate=0.0)
''' 测试阶段，只要backbone部分'''
model = load_model('convnext_tiny_hnf',args)
# model = load_model('resnet50',num_classes=6000)
#
model_path = 'Apr_21_Convnext_T_train96k_val24k_size224/'
ckpt_path = 'Model_data/'+model_path+'convnext_tiny_hnf_epoch52_T97.0286_V96.5395.pth'
print(ckpt_path)
ckpt= torch.load(ckpt_path,map_location='cpu')
model.load_state_dict(ckpt)
# new_ckpt = dict()
# for (key1, value1), (key2, value2) in zip(ckpt.items(), model.named_parameters()):
#     new_ckpt[key2] = value1
# model.load_state_dict(new_ckpt)

model = model.cuda()
model.eval()

val_file = np.load('user_data/train_data/val_2k.npy', allow_pickle=True)
valDataSet = MyDataset(train_file=val_file, mode='val', normal_num=len(val_file), num_classes=6000)
valloader = torch.utils.data.DataLoader(valDataSet, batch_size=32, shuffle=False, num_workers=4)


index  = 0
with torch.no_grad():
    accs = AverageMeter()
    model.eval()
    with torch.no_grad():
        for (inputs, labels) in tqdm(valloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            targets = labels.argmax(dim=1)
            outputs = model(inputs)
            acc = accuracy(outputs, targets)
            accs.update(acc[0].item(), inputs.size(0))
    torch.cuda.empty_cache()

print(accs.avg)
