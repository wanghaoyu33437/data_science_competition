import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

from models.densenet import densenet121
from utils import load_model
from models.googlenet import *
from models.resnet import *
from models.vgg import *
from models.mobilenet import  *
from models.mobilenetv2 import *
from models.senet import *
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform,current_path):
        images = np.load(os.path.join(current_path,'data_adv_3w_min003.npy'))
        labels = np.load(os.path.join(current_path,'label_adv_3w_min003.npy'))
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)

modelnames=['GoogLeNet','VGG16','MobileNetV2','ResNet18','SENet18']
def loadModels(modelnames):
    models=[]
    for name in modelnames:
        if name=='VGG16':
            model=eval("VGG('VGG16')")
        else:
            model=eval(name+'()')
        ckpt = torch.load('./target_models2/'+name+'_ckpt.t7')
        print(name+'\' acc :'+str(ckpt['acc']))
        model.load_state_dict(ckpt['net'])
        model = model.cuda()
        model.eval()
        models.append(model)
    return models

# models=loadModels(modelnames)

current_path='AllTestImageAndModel'
models=[]
resnet=load_model('resnet50')
resnet.load_state_dict(torch.load(current_path+'/resnet50.pth.tar')['state_dict'])
resnet.eval()
resnet=resnet.cuda()
models.append(resnet)
densenet=densenet121()
ckpt=torch.load(current_path+'/densenet121.pth.tar')
densenet.load_state_dict(ckpt['state_dict'])
densenet.eval()
densenet=densenet.cuda()
models.append(densenet)


mean=(0.4914, 0.4822, 0.4465)
std=(0.2023, 0.1994, 0.2010)
# normalize=Normalize(mean,std)
normalize=transforms.Normalize(mean=mean,std=std)
transform_test = transforms.Compose([
    # transforms.RandomAffine(degrees=(30, 30), translate=(0.2, 0.2)),
    #
    # transforms.GaussianBlur(3, 0.5),
    # transforms.RandomVerticalFlip(0.5),
    # transforms.RandomRotation(30),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
augment = transforms.Compose([
    transforms.RandomErasing(1,value='random'),
    transforms.RandomRotation(30)
])
data = MyDataset(transform_test,'ADVImage')


#
# data=torchvision.datasets.CIFAR10(root='./data',train=False
#                                   ,download=True,transform=transform_test)
test_loader=torch.utils.data.DataLoader(data,batch_size=64)

total=0

correct= [0]*len(models)
num=[0]*10
for image,target in tqdm(test_loader):
    image=image.cuda()
    target=target.cuda()
    total += target.shape[0]
    for i,model in enumerate(models):
        output=model(normalize(image))
        _,pred=output.max(1)
        # total+=target.shape[0]
        correct[i]+=pred.eq(target.argmax(1)).sum().item()
        # if pred.detach().cpu().item()!=   target.argmax().detach().cpu().item():
        #     num[target.argmax().detach().cpu().item()] += 1
    if total >=10000:
        correct = np.array(correct)
        print('acc : ', correct / total)
        total=0
        correct = [0]*len(models)


correct= np.array(correct)
print('acc : ',correct/total)


