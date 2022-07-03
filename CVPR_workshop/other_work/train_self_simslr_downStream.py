import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from fastai.vision.all import *
# from fastai.callback.wandb import WandbCallback
# import wandb
import torch
import torch.backends
torch.backends.cudnn.benchmark = True

from self_supervised.augmentations import *
from self_supervised.layers import *
from self_supervised.vision.moco import *
from self_supervised.vision.simclr import *

optdict = dict(sqr_mom=0.99,mom=0.95,beta=0.,eps=1e-4)
opt_func = partial(ranger, **optdict)

size,bs = 224,256
def get_dls(size, bs, workers=None):

    files = get_image_files('../dataset/smi_data/')
    splits=RandomSplitter(valid_pct=0.1)(files)
    item_aug = [RandomResizedCrop(size, min_scale=1), FlipItem(0.5)]
    tfms = [[PILImage.create, ToTensor, *item_aug],
            [parent_label, Categorize()]]

    dsets = Datasets(files, tfms=tfms, splits=splits)

    batch_tfms = [IntToFloatTensor, Normalize.from_stats(*imagenet_stats)]
    dls = dsets.dataloaders(bs=bs, num_workers=workers, after_batch=batch_tfms)
    return dls
def split_func(m): return L(m[0], m[1]).map(params)


def create_learner(size=size, arch='resnet50', encoder_path="models/simclr_sz224_epc50_encoder.pth"):
    dls = get_dls(size, bs=bs // 2)
    pretrained_encoder = torch.load(encoder_path)
    encoder = create_encoder(arch, pretrained=False, n_in=3)
    encoder.load_state_dict(pretrained_encoder)
    nf = encoder(torch.randn(2, 3, 224, 224)).size(-1)
    classifier = create_cls_module(nf, dls.c)
    model = nn.Sequential(encoder, classifier)
    learn = Learner(dls, model, opt_func=opt_func, splitter=split_func,
                    metrics=[accuracy, top_k_accuracy], loss_func=LabelSmoothingCrossEntropy())
    return learn

def finetune(size, epochs, arch, encoder_path, lr=1e-2, wd=1e-2):
    learn = create_learner(size, arch, encoder_path)
    learn.unfreeze()
    learn.fit_flat_cos(epochs, lr, wd=wd)

    final_acc = learn.recorder.values[-1][-2]
    torch.save(learn.model.state_dict(),'models/simslr_model_%.4f.pth'%(final_acc))
    return final_acc
if __name__ == '__main__':
    acc = []
    runs = 5
    for i in range(runs): acc += [finetune(size, epochs=5, arch='resnet50', encoder_path=f'models/simclr_sz224_epc100_encoder.pth')]