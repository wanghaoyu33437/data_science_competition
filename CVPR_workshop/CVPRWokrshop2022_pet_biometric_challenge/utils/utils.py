import torch
import os
# from models import *
import torch.nn.functional as F
from torch import nn
import shutil
import numpy as np
import timm
def load_model(model_arch,args):
    model = nn.DataParallel(timm.create_model(model_name=model_arch,**args))
    model.eval()
    return model
def frezze_model_withoutlayer(model,layer):
    if isinstance(model,nn.DataParallel):
        for n, p in model.module.named_parameters():
            if not n.startswith(layer):
                p.requires_grad_(False)
    else:
        for n, p in model.named_parameters():
            if not n.startswith(layer):
                p.requires_grad_(False)
def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def cos_simi(emb_target_img, emb_before_pasted,OnlyOne=False):
    if OnlyOne:
        return torch.sum(torch.mul(emb_target_img, emb_before_pasted))/ emb_target_img.norm() / emb_before_pasted.norm()
    else:
        return torch.sum(torch.mul(emb_target_img, emb_before_pasted), dim=1)/ emb_target_img.norm(dim=1) / emb_before_pasted.norm(dim=1)
# def load_model(arch,resume=False,path=None):
#
#     model = globals()[arch]()
#     if resume:
#         ckpt = torch.load(os.path.join(path,arch+'.pth.tar'))
#         model.load_state_dict(ckpt['state_dict'])
#     model.eval()
#     return model

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count