import cv2
import os
from torch.utils.data import DataLoader
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image,ImageChops, ImageEnhance
from functools import partial
from torch.nn.modules.loss import _Loss
from typing import Optional


BINARY_MODE: str = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
    def forward(self, logits, targets):
        num = targets.size(0)
        # 为了防止除0的发生
        smooth = 1
        # probs = F.sigmoid(logits)
        m1 = logits.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1706.05721.pdf
        """
        super(TverskyLoss, self).__init__()
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = 1
        self.beta = 1

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        y_true_pos = x.view(shp_x, -1)
        y_pred_pos = y.view(shp_x,-1)
        tp = (y_true_pos * y_pred_pos).sum(1)
        fn = (y_true_pos * (1 - y_pred_pos)).sum(1)
        fp = ((1 - y_true_pos) * y_pred_pos).sum(1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        # if not self.do_bg:
        #     if self.batch_dice:
        #         tversky = tversky[1:]
        #     else:
        #         tversky = tversky[:, 1:]
        tversky =1- tversky.mean()
        return tversky


class FocalLoss(_Loss):

    def __init__(
            self,
            mode: str,
            alpha: Optional[float] = None,
            gamma: Optional[float] = 2.,
            ignore_index: Optional[int] = None,
            reduction: Optional[str] = "mean",
            normalized: bool = False,
            reduced_threshold: Optional[float] = None,
    ):
        """Compute Focal loss

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            alpha: Prior probability of having positive value in target.
            gamma: Power factor for dampening weight (focal strength).
            ignore_index: If not None, targets may contain values to be ignored.
                Target values equal to ignore_index will be ignored from loss computation.
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
            reduced_threshold: Switch to reduced focal loss. Note, when using this mode you should use `reduction="sum"`.

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == MULTICLASS_MODE:

            num_classes = y_pred.size(1)
            loss = 0

            # Filter anchors with -1 label from loss computation
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            for cls in range(num_classes):
                cls_y_true = (y_true == cls).long()
                cls_y_pred = y_pred[:, cls, ...]

                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]

                loss += self.focal_loss_fn(cls_y_pred, cls_y_true)
        return loss
def focal_loss_with_logits(
        output: torch.Tensor,
        target: torch.Tensor,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        reduction: str = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
        eps: float = 1e-6,
) -> torch.Tensor:
    target = target.type(output.type())
    logpt = F.binary_cross_entropy(output, target, reduction="none")
    pt = torch.exp(-logpt)
    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt
    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)
    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor
    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)
    return loss

def metric(premask, groundtruth):
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    # if np.sum(cross) + np.sum(union) == 0:
    #     iou = 1
    return f1, iou


def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
# Decompose the "big" image into several "small" patches
def decompose(s2_path):
    path = s2_path + 'test/'
    flist = sorted(os.listdir(path))
    size_list = [512, 768, 1024]
    for size in size_list:
        path_out = s2_path + 'test_decompose_' + str(size) + '/'
        rm_and_make_dir(path_out)
    rtn_list = [[], [], [], []]
    for file in flist:
        img = cv2.imread(path + file)
        H, W, _ = img.shape
        size_idx = 0
        while size_idx < len(size_list) - 1:
            if H < size_list[size_idx+1] or W < size_list[size_idx+1]:
                break
            size_idx += 1
        rtn_list[size_idx].append(file)
        size = size_list[size_idx]
        path_out = s2_path + 'test_decompose_' + str(size) + '/'
        X, Y = H // size + 1, W // size + 1
        idx = 0
        for x in range(X-1):
            for y in range(Y-1):
                img_tmp = img[x * size: (x + 1) * size, y * size: (y + 1) * size, :]
                cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
                idx += 1
            img_tmp = img[x * size: (x + 1) * size, -size:, :]
            cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
            idx += 1
        for y in range(Y - 1):
            img_tmp = img[-size:, y * size: (y + 1) * size, :]
            cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
            idx += 1
        img_tmp = img[-size:, -size:, :]
        cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
        idx += 1
    return rtn_list


# Merge the predicted images
def forensics_test_merge(split_list, s2_path, path_in, path_out, size):
    rm_and_make_dir(path_out)
    for file in split_list:
        img = cv2.imread(s2_path + 'test/' + file)
        H, W, _ = img.shape
        X, Y = H // size + 1, W // size + 1
        idx = 0
        rtn = np.zeros((H, W, 3), dtype=np.uint8)
        for x in range(X-1):
            for y in range(Y-1):
                img_tmp = cv2.imread(path_in + file[:-4] + '_%03d.png' % idx)
                rtn[x * size: (x + 1) * size, y * size: (y + 1) * size, :] = img_tmp
                idx += 1
            img_tmp = cv2.imread(path_in + file[:-4] + '_%03d.png' % idx)
            rtn[x * size: (x + 1) * size, -size:, :] = img_tmp
            idx += 1
        for y in range(Y - 1):
            img_tmp = cv2.imread(path_in + file[:-4] + '_%03d.png' % idx)
            rtn[-size:, y * size: (y + 1) * size, :] = img_tmp
            idx += 1
        img_tmp = cv2.imread(path_in + file[:-4] + '_%03d.png' % idx)
        rtn[-size:, -size:, :] = img_tmp
        idx += 1
        cv2.imwrite(path_out + file[:-4] + '.png', rtn)


def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = filename.split('.')[0] + '_resaved.jpg'
    ELA_filename = filename.split('.')[0] + '.ela.png'

    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)

    ela_im = ImageChops.difference(im, resaved_im)
    os.remove(resaved_filename)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    return ela_im

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



# def rand_bbox(size, lam):
#     W = size[3]
#     H = size[2]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = np.int(W * cut_rat)
#     cut_h = np.int(H * cut_rat)
#
#     # uniform
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)
#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)
#
#     return bbx1, bby1, bbx2, bby2