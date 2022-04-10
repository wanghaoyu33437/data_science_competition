import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import numpy as np
import torchvision
import random

from models.googlenet import *
from models.resnet import *
from models.vgg import *
from models.senet import SENet18
from models.mobilenet import  *
from models.mobilenetv2 import *
from models.senet import *
from models.densenet import densenet121
from PIL import Image
from Bag_trick.utils import *
import numpy as np
from tqdm import tqdm
from torchvision import  transforms
import torch
import torch.utils.data


torch.backends.cudnn.benchmark=True

upper_limit, lower_limit = 1,0
epsilon2 = (2 / 255.)
epsilon4 = (4 / 255.)
epsilon6 = (6 / 255.)
epsilon8 = (8 / 255.)
epsilon10 = (10 / 255.)
epsilon12 = (12 / 255.)
test_epsilon = (8 / 255.)
pgd_alpha = (1 / 255.)
test_pgd_alpha = (2 / 255.)
mean=(0.4914, 0.4822, 0.4465)
std=(0.2023, 0.1994, 0.2010)
# normalize=Normalize(mean,std)
normalize=transforms.Normalize(mean=mean,std=std)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load(os.path.join('train_data_mixup_1w','data.npy'))
        labels = np.load(os.path.join('train_data_mixup_1w','label.npy'))
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

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def labelSmoothing(target,smoothing,classes=10,dim=-1):
    confidence = 1.0 - smoothing
    true_dist=torch.zeros_like(torch.tensor(np.random.random((target.shape[0], 10))))
    true_dist.fill_(smoothing / (classes - 1))
    true_dist.scatter_(1, target.detach().cpu().unsqueeze(1), confidence)
    return true_dist


def attack_pgd(models, X, y, epsilon, alpha, attack_iters=20,norm='l_inf',use_adaptive=False, s_HE=15):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    alpha = epsilon / attack_iters
    for _ in range(1):
        # early stop pgd counter for each x
        # initialize perturbation
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True

        # craft adversarial examples
        for _ in range(attack_iters):
            loss_list=[]
            for model in models:
                output = model(normalize(X + delta))
            # if use early stop pgd
                index = slice(None,None,None)
                if use_adaptive:
                    loss = F.cross_entropy(s_HE * output, y)
                    loss_list.append(loss)
                else:
                    loss = F.cross_entropy(output, y)
                    loss_list.append(loss)
            loss=torch.mean(torch.stack(loss_list))
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(normalize(X + delta)), y, reduction='none')
        # for model in models[:-1]:
        #     all_loss += F.cross_entropy(model(normalize(X + delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

transform=transforms.Compose([
    # transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
])
# addPepperNoise = AddPepperNoise(0.6,1)

# CropFive  = transforms.FiveCrop(20)
# CropFiveResize = transforms.Resize(32)
augment = transforms.Compose([
    transforms.RandomErasing(1,value='random'),
    transforms.RandomRotation(30)
])
# randomErasing=transforms.RandomErasing(1,value='random')
# randomRotation=transforms.RandomRotation(30)
# randomColor=transforms.ColorJitter(0.5,0.5,0.5)

dataset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
test_batches = torch.utils.data.DataLoader(dataset,batch_size=64,num_workers=4)
# train_set = list(zip(transpose(pad(dataset['train']['data'][:5000], 4) / 255.),
#                      dataset['train']['labels'][:5000]))

# train_set_x = Transform(train_set, transforms)
# train_batches = Batches(train_set_x,64, shuffle=True, set_random_choices=True, num_workers=4)
# dataset=MyDataset(transform=transform)
# trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

modelnames=['GoogLeNet','MobileNetV2','ResNet18','SENet18']
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

models=loadModels(modelnames)

#
images = []
inputs_list=[]
soft_labels = []
hard_labels = []
total=0
num = [0]*10
correct= [0]*len(models)
for inputs,target in tqdm(test_batches):
    # 切成五份
    # one_hot = torch.zeros_like(torch.tensor(np.random.random((target.shape[0], 10))))
    # one_hot.scatter_(1, target.unsqueeze(1), 1.)
    # # SmoothingLabel = labelSmoothing(target,0.2)
    # one_hot = one_hot.numpy()
    # SmoothingLabel = SmoothingLabel.numpy()
    # inputs_numpy=inputs.numpy()
    # img = np.uint8(transpose(inputs_numpy * 255, 'NCHW', 'NHWC'))
    # images.extend(img)
    # hard_labels.extend(SmoothingLabel)
    # hard_labels.extend(one_hot)
    # inputs_aug_numpy = augment(inputs).numpy()
    # images.extend( np.uint8(transpose(inputs_aug_numpy * 255, 'NCHW', 'NHWC')))
    # if random.randint(0,9)<5:
    inputs=inputs.cuda()
    target = target.cuda()
# # 对抗样本
    delta = attack_pgd(models, inputs, target, norm='l_inf',epsilon=(30/255), alpha=pgd_alpha, attack_iters=10)
    adv_input = torch.clamp(inputs + delta[:inputs.size(0)], min=lower_limit, max=upper_limit)
    adv_input_numpy=adv_input.detach().cpu().numpy()
    adv_img = np.uint8(transpose(adv_input_numpy * 255, 'NCHW', 'NHWC'))
    images.extend(adv_img)
#     hard_labels.extend(one_hot)
    # for i,model in enumerate(models):
    #     output=model(adv_input)
    #     _,pred=output.max(1)
    #     total+=target.shape[0]
    #     correct[i]+=pred.eq(target).sum().item()
    # hard_labels.extend(SmoothingLabel)
    # adv_aug_numpy = augment(adv_input.detach().cpu()).numpy()
    # images.extend(np.uint8(transpose(adv_aug_numpy * 255, 'NCHW', 'NHWC')))
    # hard_labels.extend(one_hot)
    # else :
    #     if random.randint(0,9)<6:
    #         img = addPepperNoise(img)
    #         images.extend(img)
    #         hard_labels.extend(one_hot)
#
# #
#


images = np.array(images)
hard_labels  = np.array( hard_labels)
print(images.shape, images.dtype, hard_labels.shape, hard_labels.dtype)
current_path = 'ADVImage'

if not os.path.exists(current_path):
    os.mkdir(current_path)
np.save(os.path.join(current_path,'data30.npy'), images)
# np.save(os.path.join(current_path,'label.npy'), hard_labels)