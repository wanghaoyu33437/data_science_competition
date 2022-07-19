from torch import nn
import torch
import torch.nn.functional as F
class CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''

    def __init__(self, label_smooth=None, class_num=4):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target,soft_label=None):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12
        if soft_label is not None:
            logprobs = F.log_softmax(pred, dim=1)
            soft_label = F.softmax(soft_label,dim=1)
            loss = -1 * torch.sum(logprobs*soft_label,1)

        elif self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)  # softmax + log
            target = F.one_hot(target, self.class_num)  # 转换成one-hot

            # label smoothing
            # 实现 1
            # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num
            # 实现 2
            # implement 2
            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1),
                                 max=1.0 - self.label_smooth)
            loss = -1 * torch.sum(target * logprobs, 1)

        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))
        return loss.mean()
        nn.CrossEntropyLoss
