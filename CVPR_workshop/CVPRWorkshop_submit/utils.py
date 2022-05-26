import os

import shutil

import timm
def load_model(model_arch,arg):
    model = timm.create_model(model_name=model_arch,**arg)
    model.eval()
    return model

def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

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


import torch
def get_best_f1(predicts,labels):
    BestF1 = 0
    ths_sort, _ = torch.sort(predicts, 0)
    best_th = 0
    BestTP,BestFP,BestTN,BestFN=0,0,0,0
    for th in ths_sort:
        TP = ((predicts >= th) & (labels == 1)).cpu().numpy().sum()
        FP = ((predicts >= th) & (labels == 0)).cpu().numpy().sum()
        TN = ((predicts <= th) & (labels== 0)).cpu().numpy().sum()
        FN = ((predicts <= th) & (labels == 1)).cpu().numpy().sum()
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
        if F1 > BestF1:
            best_th = th
            BestF1 = F1
            BestTP,BestFP,BestTN,BestFN=TP,FP,TN,FN
    return BestF1,BestTP,BestFP,BestTN,BestFN,best_th