import torchattacks
from utils import *
from torch import nn
from torchvision import transforms
# from utils import AverageMeter, accuracy
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data
import cv2
import os
import argparse

torch.backends.cudnn.benchmark=True
lower_limit = 0
upper_limit = 1

train_size = 224


class Normalize(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        input = input
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,data_path):
        self.num= len(train_file)
        self.train_file = train_file
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __getitem__(self, index):
        return self.load_item(index)

    def __len__(self):
        return self.num
    def load_item(self,index):
        image_path,label= self.train_file[index]
        image_name = image_path.split('/')[-1]
        image = cv2.imread(image_path)[:,:,::-1]
        image = cv2.resize(image,(train_size,train_size))
        label = label.astype(np.float32)
        image = self.transform(image)
        return image,label,image_name

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)



attack_params = {
    'Square':{
        'eps':8/255,
        'n_queries':500
    },
    'DeepFool':{
        'overshoot':0.1
    },
    'PGD':{
        'eps':5./255.,
        'alpha':5./10./255.,
        'steps':20,
    }
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--attack',type=str,choices=['Square','DeepFool','PGD'])
    parser.add_argument('--GPU',type=str,default='2')
    parser.add_argument('--batch_size',type=int,default=64)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    train_file = np.load('train_data/clean/clean.npy', allow_pickle=True)
    TrainDataSet = MyDataset(train_file)
    trainloader = torch.utils.data.DataLoader(TrainDataSet, batch_size=args.batch_size, shuffle=False, num_workers=4)

    resnet50_args = {'pretrained': False, 'num_classes': 20}
    resnet50 = nn.DataParallel(load_model('resnet50', resnet50_args))
    resnet50.load_state_dict(torch.load('./model_weight/resnet50_clean.pth', map_location='cpu'))
    model = nn.Sequential(
        Normalize(),
        resnet50,
    )
    model = model.cuda()
    model.eval()

    attack = torchattacks.__dict__[args.attack](model=model,**attack_params[args.attack])

    data_adv_paths= []
    output_dir = 'dataset/adv/%s/'%args.attack
    for inputs,target,names in tqdm(trainloader):
        inputs=inputs.cuda()
        target = target.argmax(1).cuda()
        # target = target.cuda()
        # with torch.no_grad():
        #     output = model(inputs)
        #     acc = accuracy(output, target)
        # print('Clean image acc %f'%acc[0].item())
        # # 对抗样本
        adv_img = attack(inputs,target)
        adv_input = torch.clamp(adv_img, min=lower_limit, max=upper_limit)
        # Attack ACC
        # with torch.no_grad():
        #     output = model(adv_input)
        #     acc = accuracy(output, target)
        # print(acc[0].item())

        adv_input_numpy=adv_input.detach().cpu().numpy()
        adv_img = np.uint8(adv_input_numpy.transpose([0,2,3,1])* 255)[:,:,:,::-1]
    #     # break
        os.makedirs(output_dir,exist_ok=True)
        for img, _,n in zip(adv_img, target.cpu().numpy(),names):
            cv2.imwrite(output_dir + n[:-5] + '.png', img.astype(np.uint8))
            data_adv_paths.append([output_dir + n[:-5] + '.png', 1])
    np.save('train_data/adv/%s.npy'%args.attack, data_adv_paths)
