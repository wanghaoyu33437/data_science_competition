import numpy as np
import torchvision
import random
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
# from torchvision import transforms
from models.googlenet import *
from models.resnet import *
from models.vgg import *
from models.mobilenet import  *
from models.mobilenetv2 import *
from models.senet import *
from models.densenet import densenet121

from Bag_trick.utils import *
import numpy as np
from tqdm import tqdm

torch.backends.cudnn.benchmark=True
device=torch.device('cuda:0')
upper_limit, lower_limit = 1,0
epsilon = (8 / 255.)
epsilon6 = (6 / 255.)
epsilon8 = (8 / 255.)
epsilon10 = (10 / 255.)
epsilon12 = (12 / 255.)
test_epsilon = (8 / 255.)
pgd_alpha = (2 / 255.)
test_pgd_alpha = (2 / 255.)
mean=(0.4914, 0.4822, 0.4465)
std=(0.2023, 0.1994, 0.2010)
normalize=Normalize(mean,std)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def labelSmoothing(target,smoothing,classes=10,dim=-1):
    confidence = 1.0 - smoothing
    true_dist=torch.zeros_like(torch.tensor(np.random.random((target.shape[0], 10))))
    true_dist.fill_(smoothing / (classes - 1))
    true_dist.scatter_(1, target.detach().cpu().unsqueeze(1), confidence)
    return true_dist


def attack_pgd(models, X, y, epsilon, alpha, attack_iters=10,norm='l_inf',use_adaptive=False, s_HE=15):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()

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
                output = model(X + delta)
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

        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    return max_delta
def eps_alpha_schedule(t, warm_up_eps = False, if_use_stronger_adv=False, stronger_index=0): # Schedule number 0
    if stronger_index == 0:
        epsilon_s = [epsilon * 1.5, epsilon * 2]
        pgd_alpha_s = [pgd_alpha, pgd_alpha]
    elif stronger_index == 1:
        epsilon_s = [epsilon * 1.5, epsilon * 2]
        pgd_alpha_s = [pgd_alpha * 1.25, pgd_alpha * 1.5]
    elif stronger_index == 2:
        epsilon_s = [epsilon * 2, epsilon * 2.5]
        pgd_alpha_s = [pgd_alpha * 1.5, pgd_alpha * 2]
    else:
        print('Undefined stronger index')

    if if_use_stronger_adv:
        if t < 100:
            if t < 15 and warm_up_eps:
                return (t + 1.) /15 * epsilon, pgd_alpha, 1
            else:
                return epsilon, pgd_alpha,1
        elif t < 105:
            return epsilon_s[0], pgd_alpha_s[0],1
        else:
            return epsilon_s[1], pgd_alpha_s[1], 1
    else:
        if t < 15 and warm_up_eps:
            return (t + 1.) /15 * epsilon, pgd_alpha, 1
        else:
            return epsilon, pgd_alpha, 1



transforms = [Crop(32, 32), FlipLR()]
dataset = cifar10('./data')
test_set = list(zip(transpose(pad(dataset['test']['data'], 4) / 255.),
                     dataset['test']['labels']))
test_set_x = Transform(test_set, transforms)
test_batches = Batches(test_set_x,64, shuffle=True, set_random_choices=True, num_workers=4)
normalize1=torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# train_set = list(zip(transpose(pad(dataset['train']['data'][:5000], 4) / 255.),
#                      dataset['train']['labels'][:5000]))
# train_set_x = Transform(train_set, transforms)
# train_batches = Batches(train_set_x,64, shuffle=True, set_random_choices=True, num_workers=4)

# models=[]
# googlenet=GoogLeNet()
# resnet=ResNet18()
# vgg=VGG('VGG16')
# mobilenet=MobileNetV2()
#
# #
# ckpt=torch.load('./target_models/GoogLeNet_ckpt.t7')
# print(ckpt['acc'])
# googlenet.load_state_dict(ckpt['net'])
# googlenet=googlenet.to(device)
# googlenet.eval()
# models.append(googlenet)
#
# ckpt=torch.load('./target_models/VGG16_ckpt.t7')
# print(ckpt['acc'])
# vgg.load_state_dict(ckpt['net'])
# vgg=vgg.to(device)
# vgg.eval()
# models.append(vgg)
#
# ckpt=torch.load('./target_models/MobileNetV2_ckpt.t7')
# print(ckpt['acc'])
# mobilenet.load_state_dict(ckpt['net'])
# mobilenet=mobilenet.to(device)
# mobilenet.eval()
# models.append(mobilenet)
#
# ckpt=torch.load('./target_models/ResNet_ckpt.t7')
# print(ckpt['acc'])
# resnet.load_state_dict(ckpt['net'])
# resnet=resnet.to(device)
# resnet.eval()
# models.append(resnet)



images = []
soft_labels = []
hard_labels = []
total=0
for batch in tqdm(test_batches):
    inputs=batch['input']
    target=batch['target']
    # 先保存正常样本 再保存一批对抗样本128个保存一批
    inputs_numpy=inputs.detach().cpu().numpy()
    img = np.uint8(transpose(inputs_numpy * 255, 'NCHW', 'NHWC'))
    images.extend(img)
    # one_hot = labelSmoothing(target,smoothing=0.2)
    one_hot=torch.zeros_like(torch.tensor(np.random.random((target.shape[0], 10))))
    one_hot.scatter_(1, target.detach().cpu().unsqueeze(1), 1.)
    one_hot = one_hot.numpy()
    hard_labels.extend(one_hot)
    if random.randint(0,10)<=8:
        delta = attack_pgd(models, inputs, target, epsilon=epsilon8, alpha=pgd_alpha, attack_iters=10)
        adv_input = torch.clamp(inputs + delta[:inputs.size(0)], min=lower_limit, max=upper_limit)
        adv_input_numpy=adv_input.detach().cpu().numpy()
        adv_img = np.uint8(transpose(adv_input_numpy * 255, 'NCHW', 'NHWC'))
        images.extend(adv_img)
        hard_labels.extend(one_hot)

#
images = np.array(images)
hard_labels  = np.array( hard_labels)
print(images.shape, images.dtype, hard_labels.shape, hard_labels.dtype)
current_path = 'pgd8_test_passHardSample_1to1_ensemble'
if not os.path.exists(current_path):
    os.mkdir(current_path)
np.save(os.path.join(current_path,'data.npy'), images)
np.save(os.path.join(current_path,'label.npy'), hard_labels)