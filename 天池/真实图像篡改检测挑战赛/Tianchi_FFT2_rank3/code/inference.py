import numpy as np
import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from main import GIID_Model,GIID_Dataset
os.environ['CUDA_VISIBLE_DEVICES']='3'
import cv2
from gen_dataset import merge
from PIL import Image
from transform import TTA
from utils import metric
from torchvision import transforms
transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            # transforms.Normalize((0.87185234, 0.86747855, 0.8583319), (0.15480465, 0.16086526, 0.16299605)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
size = 896
# fold = 'images_compose_896'
# fold  = 'images_compose_768_896'
# test_path = '../../baseline/data/test_decopose_768_896/img_%d/'%size
test_path = '../../baseline/data/test_decopose_896/'
# test_path = '../../baseline/data/test/img/'
# train_path = '../../baseline/data/train/img/'
mask_path =  '../../baseline/data/train/mask/'
test_paths = sorted(os.listdir(test_path))
mask_paths =sorted(os.listdir(mask_path))
# test_dataset = GIID_Dataset(len(os.listdir(test_path)), choice='test',test_path=test_path)
# test_loader = DataLoader(test_dataset,batch_size=16,num_workers=4,pin_memory=True)
backbone_arch = 'senet154'#senet154 , seresnext50.LinkNet_se_resnet152
model = GIID_Model(backbone_arch=backbone_arch).cuda()
model_path = 'Mar_18_AllImg_s3data_2k_bookscreen_1446_size768_newDataAug_senet154_bce07_dice03_focal1/tmp_epoch_091_train1.7438_val1.1071_valLoss0.2411/'
model.load(model_path)
model.eval()
tta =TTA()

def output(model,input,ori_size,size=512):
    img = cv2.resize(input, (size, size))
    # img = input.resize((size,size))
    input = transform(img).unsqueeze(0).cuda().half()
    Mo = model(input)
    Mo = Mo * 255.
    Mo = Mo.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
    Mo = cv2.resize(Mo.astype(np.float32), (ori_size[1], ori_size[0]))
    # Mo = np.array(Image.fromarray(Mo.astype(np.uint8)[:, :, 0]).resize((ori_size[0], ori_size[1]), Image.BILINEAR))
    return Mo
save_path = '../save_out/test/'+backbone_arch+'/'+'Mar_18_'+model_path.split('/')[1][14:]+"_AllImg_s3data_2k_bookscreen_1446_size768_newDataAug_senet154_bce07_dice03_focal1/"
bad_examples=[]
with torch.no_grad():
    scores = 0
    f1, iou, gen_losses = [], [], []
    for img_name in tqdm(test_paths[:]):  # test
        # break
        # img = Image.open(test_path+img_name)
        img = cv2.imread(test_path+img_name)[:,:,::-1]
        # mask_gt = np.array(Image.open(mask_path+mask_name))
        ori_size = img.shape
        # ori_size = img.size
        img = img.astype('float')/255
        # masks = []
        ''' tta '''
        Mo = output(model, img, ori_size,size)
        Mo[Mo < 100] = 0
        Mo[Mo >= 100] = 255
        # masks.append(Mo.astype(np.uint8))
        #
        # img1 = tta.VerticalFlip(img)
        # Mo = output(model, img1, ori_size, size)
        # Mo = tta.VerticalFlip(Mo)
        # Mo[Mo < 100] = 0
        # Mo[Mo >= 100] = 255
        # masks.append(Mo.astype(np.uint8))

        # img2 = tta.HorizontalFlip(img)
        # Mo = output(model, img2, ori_size, size)
        # Mo = tta.HorizontalFlip(Mo)
        # Mo[Mo < 100] = 0
        # Mo[Mo >= 100] = 255
        # masks.append(Mo.astype(np.uint8))

        # mask_and = masks[0]
        # for i in range(1, len(masks)):
        #     mask_and = mask_and & masks[i]
        # mask_or = masks[0]
        # for i in range(1, len(masks)):
        #     mask_or = mask_or | masks[i]
        '''tta '''
        # img = cv2.resize(img, (size,size))
        # inputs = transform(img).unsqueeze(0).cuda()
        # Mo = model(inputs)
        # Mo = Mo * 255.
        # Mo = Mo.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
        # Mo = cv2.resize(Mo, (ori_size[1], ori_size[0]))
        # Mo[Mo < 100] = 0
        # Mo[Mo >= 100] = 255
        # save_seg_path = os.path.join(save_path, 'images_compose_tta_HF_OR_896', os.path.split(img_name)[-1].split('.')[0] + '.png')
        # save_seg_path1 = os.path.join(save_path, 'images_compose_tta_HF_AND_896',os.path.split(img_name)[-1].split('.')[0] + '.png')
        # break
        save_single_path = os.path.join(save_path, 'images', os.path.split(img_name)[-1].split('.')[0] + '.png')

        # if (mask_and == 255).sum()<=3000:
        #     bad_examples.append(img_name)
        # os.makedirs(os.path.split(save_seg_path)[0], exist_ok=True)
        # os.makedirs(os.path.split(save_seg_path1)[0], exist_ok=True)

        os.makedirs(os.path.split(save_single_path)[0], exist_ok=True)
        # f1, iou = metric(Mo/255,mask_gt/255)
        # scores+=(f1+iou)
        # cv2.imwrite(save_seg_path, mask_or.astype(np.uint8))
        # cv2.imwrite(save_seg_path1, mask_and.astype(np.uint8))
        cv2.imwrite(save_single_path, Mo.astype(np.uint8))
        # Image.fromarray(Mo).save(save_single_path)
        # cv2.imwrite(save_single_path, masks[0].astype(np.uint8))
size_list=[896]
merge(save_path+'images/',size_list)


    # for img,_,img_name in tqdm(test_loader):  # test
    #     # img = Image.open(test_path+img_name)
    #     img = img.cuda()
    #     Mo = model(img)
    #     Mo = Mo * 255.
    #     Mo = Mo.permute(0, 2, 3, 1).cpu().detach().numpy()
    #     Mo[Mo < 100] = 0
    #     Mo[Mo >= 100] = 255
    #     for i in range(img.size(0)):
    #         save_seg_path = os.path.join(save_path, 'images_compose', img_name[i])
    #         os.makedirs(os.path.split(save_seg_path)[0], exist_ok=True)
    #         cv2.imwrite(save_seg_path, Mo[i].astype(np.uint8))
# np.save('./'+backbone_arch+"_"+model_path.split('/')[1][-6:]+'_bad.npy',bad_examples)