import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from fastai.vision.all import *
# from fastai.callback.wandb import WandbCallback
# import wandb

torch.backends.cudnn.benchmark = True
WANDB = False

from self_supervised.augmentations import *
from self_supervised.layers import *
from self_supervised.vision.moco import *
from self_supervised.vision.simclr import *


def get_dls(size, bs, workers=None):
    files = get_image_files('../dataset/smi_data/')
    tfms = [[PILImage.create, ToTensor, RandomResizedCrop(size, min_scale=1.)],
            [parent_label, Categorize()]]

    dsets = Datasets(files, tfms=tfms, splits=RandomSplitter(valid_pct=0.2)(files))

    batch_tfms = [IntToFloatTensor]
    dls = dsets.dataloaders(bs=bs, num_workers=workers, after_batch=batch_tfms)
    return dls

bs, resize, size = 160, 256, 224
dls = get_dls(resize, bs)

arch = "resnet50"
encoder = create_encoder(arch, pretrained=False, n_in=3)

dls = get_dls(resize, bs)
K = bs*2**4
assert K < len(dls.train_ds)

# model = create_moco_model(encoder,hidden_size=2048, projection_size=5)
# aug_pipelines = get_moco_aug_pipelines(size, rotate=True, rotate_deg=10, jitter=True, bw=True, blur=False,stats=None)
# cbs=[MOCO(aug_pipelines, K=K)]
model = create_simclr_model(encoder)
aug_pipelines = get_simclr_aug_pipelines(size, rotate=True, rotate_deg=10, jitter=True, bw=True, blur=False,stats=None)
cbs=[SimCLR(aug_pipelines)]
learn = Learner(dls, model, cbs=cbs)

learn.to_fp16()
lr,wd,epochs=1e-2,1e-2,100

learn.unfreeze()
learn.fit_flat_cos(epochs, lr, wd=wd, pct_start=0.5)

# save_name = f'moco_iwang_sz{size}_epc{epochs}'
save_name = f'simclr_sz{size}_epc{epochs}'
learn.save(save_name)
torch.save(learn.model.encoder.state_dict(), learn.path/learn.model_dir/f'{save_name}_encoder.pth')