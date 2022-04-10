import numpy as np
from scipy import stats as st
import torch
import torch.nn.functional as F


def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    stack_kern = np.expand_dims(stack_kern, 1)
    stack_kern = torch.tensor(stack_kern).cuda()
    return stack_kern, kern_size // 2


def project_noise(x, stack_kern, kern_size):
    # x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
    x = F.conv2d(x, stack_kern, padding = (kern_size, kern_size), groups=3)
    return x


def input_diversity(x, resize_rate, diversity_prob):
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)

    if resize_rate < 1:
        img_size = img_resize
        img_resize = x.shape[-1]

    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

    return padded if torch.rand(1) < diversity_prob else x


def kernel_generation(kernel_name='gaussian', len_kernel=15, nsig=3):
    if kernel_name == 'gaussian':
        kernel = gkern(len_kernel, nsig).astype(np.float32)
    elif kernel_name == 'linear':
        kernel = lkern(len_kernel).astype(np.float32)
    elif kernel_name == 'uniform':
        kernel = ukern(len_kernel).astype(np.float32)
    else:
        raise NotImplementedError

    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    return stack_kernel


def gkern( kernlen=15, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def ukern( kernlen=15):
    kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
    return kernel


def lkern( kernlen=15):
    kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def torch_staircase_sign(noise, n):
    noise_staircase = torch.zeros(size=noise.shape).cuda()
    sign = torch.sign(noise).cuda()
    temp_noise = noise.cuda()
    abs_noise = abs(noise)
    base = n / 100
    percentile = []
    for i in np.arange(n, 100.1, n):
        percentile.append(i / 100.0)
    medium_now = torch.quantile(abs_noise.reshape(len(abs_noise), -1), q = torch.tensor(percentile, dtype=torch.float32).cuda(), dim = 1, keepdim = True).unsqueeze(2).unsqueeze(3)

    for j in range(len(medium_now)):
        # print(temp_noise.shape)
        # print(medium_now[j].shape)
        update = sign * (abs(temp_noise) <= medium_now[j]) * (base + 2 * base * j)
        noise_staircase += update
        temp_noise += update * 1e5

    return noise_staircase


