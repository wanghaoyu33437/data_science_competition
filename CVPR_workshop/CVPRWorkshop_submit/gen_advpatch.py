# Adversarial Patch Attack
# Created by Wwwwhy 2020/3/17
"""
Reference:
[1] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""
import torch
import torch.nn.functional as F
from gen_perturbation import Normalize
from torch import nn
import argparse
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
from torchvision import transforms
from utils import*
from tqdm import tqdm
import os
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,train_file,mode='train'):
        self.num= len(train_file)
        self.train_file = train_file
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    def __getitem__(self, index):
        return  self.load_item(index)

    def __len__(self):
        return self.num
    def load_item(self,index):
        image_path,label = self.train_file[index]
        image_name = image_path.split('/')[-1]
        image = cv2.imread(image_path)[:,:,::-1]
        label = label.astype(np.float32)
        image = self.transform(image)

        return image,label,image_name

def patch_initialization(patch_type='rectangle', image_size=(3, 224, 224), noise_percentage=0.03):
    if patch_type == 'rectangle':
        mask_length = int((noise_percentage * image_size[2] * image_size[3])**0.5)
        patch = np.random.rand(image_size[0],3,mask_length, mask_length)
    return patch
# Generate the mask and apply the patch
# TODO: Add circle type
def mask_generation(mask_type='rectangle', patch=None, image_size=(16,3, 224, 224)):
    applied_patch = np.zeros(image_size)
    if mask_type == 'rectangle':
        x_location, y_location = np.random.randint(low=0, high=image_size[3]-patch.shape[3]), np.random.randint(low=0, high=image_size[2]-patch.shape[2])
        for i in range(patch.shape[0]):
            applied_patch[i,:, y_location:y_location + patch.shape[2], x_location:x_location + patch.shape[3]] = patch[i]
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return applied_patch, mask, x_location,y_location


upper_limit, lower_limit = 1,0




def patch_attack(image, applied_patch, mask, target, model, max_iteration=100):
    model.eval()
    lr = 0.5 / max_iteration
    applied_patch = torch.from_numpy(applied_patch)
    # applied_patch.requires_grad_(True)
    mask = torch.from_numpy(mask)
    perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul(
        (1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
    perturbated_image = torch.clamp(perturbated_image, min=lower_limit, max=upper_limit)
    target_probability, count = 0, 0
    while count < max_iteration:
        count += 1
        # Optimize the patch
        perturbated_image = perturbated_image.cuda()
        perturbated_image.requires_grad_(True)
        output = model(perturbated_image)
        loss = F.cross_entropy(output,target)
        loss.backward()
        patch_grad = perturbated_image.grad.clone().cpu()
        perturbated_image.grad.data.zero_()
        applied_patch = applied_patch + lr * torch.sign(patch_grad)
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = torch.clamp(perturbated_image, min=lower_limit, max=upper_limit)
        perturbated_image = perturbated_image.cuda()

    perturbated_image = perturbated_image.clone().detach().cpu().numpy()
    applied_patch = applied_patch.cpu().numpy()
    return perturbated_image, applied_patch

# os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

# Load the model



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str, default='Patch', help="attack name")
    parser.add_argument('--noise_percentage', type=float, default=0.02, help="percentage of the patch size compared with the image size")
    parser.add_argument('--probability_threshold', type=float, default=0.3, help="minimum target probability")

    parser.add_argument('--max_iteration', type=int, default=70, help="max iteration")
    parser.add_argument('--epochs', type=int, default=20, help="total epoch")
    parser.add_argument('--patch_type', type=str, default='rectangle', help="type of the patch")
    parser.add_argument('--GPU', type=str, default='0', help="index pf used GPU")
    parser.add_argument('--batch_size',type=int,default=64)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

    resnet50_args = {'pretrained': False, 'num_classes': 20}
    resnet50 = nn.DataParallel(load_model('resnet50', resnet50_args))
    resnet50.load_state_dict(torch.load('./model_weight/resnet50_clean.pth', map_location='cpu'))
    model = nn.Sequential(
        Normalize(),
        resnet50,
    )
    model = model.cuda()
    model.eval()
    new_ckpt = dict()

    best_patch_epoch, best_patch_success_rate = 0, 0

    train_file = np.load('train_data/clean/clean.npy', allow_pickle=True)
    TrainDataSet = MyDataset(train_file)
    trainloader = torch.utils.data.DataLoader(TrainDataSet, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_total, train_actual_total, train_success = 0, 0, 0
    img_name  = 0
    data_label_paths = []
    output_dir = 'dataset/adv/%s/'%args.attack
    for (image, label,names) in tqdm(trainloader):
        train_total += label.shape[0]
        label_numpy = label.numpy()
        image = image.cuda()
        label = label.cuda()
        target = label.argmax(1)
        with torch.no_grad():
            output = model(image)
            acc = accuracy(output, target)
        print('Clean image acc %f'%acc[0].item())

        patch = patch_initialization(args.patch_type, image_size=image.shape, noise_percentage=args.noise_percentage)
        applied_patch, mask, x_location, y_location = mask_generation(args.patch_type, patch, image_size=image.shape)
        patch_image, applied_patch = patch_attack(image, applied_patch, mask, target, model, args.max_iteration)
        # break
        # output = model(normalize(torch.from_numpy(perturbated_image).cuda()))
        with torch.no_grad():
            output = model(torch.from_numpy(patch_image).cuda())
        acc =accuracy(output,target)

        print(acc[0].item())

        perturbated_image =(patch_image.transpose([0,2,3,1])*255.0)[:,:,:,::-1]
        os.makedirs(output_dir, exist_ok=True)
        for img, n in zip(perturbated_image,names):
            cv2.imwrite(output_dir + n[:-5] + '.png', img.astype(np.uint8))
            data_label_paths.append([output_dir + n[:-5] + '.png', 1])
    np.save('train_data/adv/%s.npy'%args.attack, data_label_paths)
