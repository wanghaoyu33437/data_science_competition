import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

from fastai.vision.all import *
from self_supervised.layers import *
from self_supervised.vision.simclr import *
from self_supervised.augmentations import *
import torch

import torch.backends.cudnn

torch.backends.cudnn.benchmark =True

def get_dls(size, bs, workers=None):
    files = get_image_files('dataset/pet_biometric_challenge_2022/class_image/train/')
    tfms = [[PILImage.create, ToTensor, RandomResizedCrop(size, min_scale=1.)],
            [parent_label, Categorize()]]

    dsets = Datasets(files, tfms=tfms, splits=RandomSplitter(valid_pct=0.1)(files))

    batch_tfms = [IntToFloatTensor]
    dls = dsets.dataloaders(bs=bs, num_workers=workers, after_batch=batch_tfms)
    return dls


bs,resize,size = 64,400,368
dls = get_dls(resize, bs)

encoder = create_encoder("convnext_tiny_hnf", n_in=3, pretrained=True)
model = create_simclr_model(encoder, hidden_size=2048, projection_size=6000)
# model = nn.DataParallel(model)
aug_pipelines = get_simclr_aug_pipelines(size=size, rotate=True, jitter=True, bw=True, blur=True, rotate_deg=10,blur_s=(4,16), blur_p=0.25,stats=None)
learn = Learner(dls, model,cbs=[SimCLR(aug_pipelines, temp=0.07, print_augs=True)])

# b = dls.one_batch()
# # learn._split(b)
# # learn('before_batch')
# # learn.sim_clr.show(n=5)

learn.to_fp16()
lr,wd,epochs=1e-3,5e-2,50
learn.unfreeze()
learn.fit_flat_cos(epochs, lr, wd=wd, pct_start=0.5)

save_name = f'simclr_sz{size}_epc{epochs}'
learn.save(save_name)
torch.save(learn.model.encoder.state_dict(), learn.path/learn.model_dir/f'{save_name}_encoder.pth')
learn.recorder.plot_loss()