import os
import numpy as np
from models import irse, facenet, mobileFacenet
import torch.nn.functional as F
import torch.backends.cudnn
from utils import clip_by_tensor, cos_simi, read_img_from_path, conv2d_same_padding, cv2_save,compareIm
from msssim import msssim
from attack_method import input_diversity, kernel_generation, torch_staircase_sign, project_noise, project_kern

import argparse

torch.backends.cudnn.benchmark = True

def set_model_info(test_model_list,device):
    models_info = dict()
    print(test_model_list)
    for model_name in test_model_list:
        models_info[model_name] = [[], []]
        if model_name == 'ir50_ms1m':
            models_info[model_name][0].append((112, 112))
            fr_model = irse.IR_50((112, 112))
            fr_model.load_state_dict(torch.load('./user_data/models/backbone_ir50_ms1m_epoch120.pth'))
        if model_name == 'IR_50_LFW_AdvTrain':
            models_info[model_name][0].append((112, 112))
            fr_model = irse.IR_50((112, 112))
            fr_model.load_state_dict(torch.load('./user_data/models/Backbone_IR_50_LFW_ADV_TRAIN.pth'))
        if model_name == 'IR_101':
            models_info[model_name][0].append((112, 112))
            fr_model = irse.IR_101((112, 112))
            fr_model.load_state_dict(torch.load('./user_data/models/Backbone_IR_101_Batch_108320.pth'))
        # 以上是OPPO给的官方models,返回人脸识别模型（主干网络）（输出512维的向量）
        if model_name == 'ir152':
            models_info[model_name][0].append((112, 112))
            fr_model = irse.IR_152((112, 112))
            fr_model.load_state_dict(torch.load('./user_data/models/ir152.pth'))
        if model_name == 'irse50':
            models_info[model_name][0].append((112, 112))
            # fr_model = irse.Backbone(50, 0.6, 'ir_se')
            fr_model = irse.IR_SE_50((112, 112))
            fr_model.load_state_dict(torch.load('./user_data/models/irse50.pth'))
        if model_name == 'mobile_face':
            models_info[model_name][0].append((112, 112))
            fr_model = mobileFacenet.MobileFaceNet(512)
            fr_model.load_state_dict(torch.load('./user_data/models/mobile_face.pth'))
        if model_name == 'facenet':
            models_info[model_name][0].append((160, 160))
            fr_model = facenet.InceptionResnetV1(num_classes=8631, device=device)
            fr_model.load_state_dict(torch.load('./user_data/models/facenet.pth'))
        fr_model.eval()
        fr_model.to(device)
        models_info[model_name][0].append(fr_model)
    return models_info

def graph(x,com_x,settings, models_info, which):
    # parameters setting
    eps = settings.max_epsilon / 255.0
    steps = settings.steps
    mu=1
    beta = 0.5
    alpha = eps / steps
    alpha_beta = alpha * settings.amplification
    th=settings.threshold
    decay = settings.momentum
    momentum = torch.zeros_like(x).detach().to(device)
    resize_rate = settings.resize_rate
    resize_chance = settings.resize_chance
    amplification = 0.0

    gaussian_kernel = torch.from_numpy(kernel_generation()).to(device)
    stack_kern, kern_size = project_kern(3)

    adv = x.clone()
    adv = adv.to(device)
    print('Eps:',settings.max_epsilon ,'MSSSIM :',th)
    for i in range(steps+20):
        adv.requires_grad = True
        com_loss_list=list()
        for j in range(resize_rate.shape[-1]):
            for model_name in['ir152','ir50_ms1m',  'mobile_face', 'facenet', 'irse50','IR_50_LFW_AdvTrain','IR_101']:
                embbeding = \
                    models_info[model_name][0][1](
                                F.interpolate(
                                input_diversity(torch.cat((adv,F.interpolate(com_x,size=adv.size()[-1]))), resize_rate=resize_rate[j], diversity_prob=resize_chance),
                                size=models_info[model_name][0][0], mode='bilinear')
                        )
                adv_embbeding,com_embeddings=embbeding[0:1],embbeding[1:2]
                if model_name =='ir152' or model_name=='facenet':
                    com_loss_list.append((cos_simi(adv_embbeding, com_embeddings)))
                else:
                    com_loss_list.append((2*cos_simi(adv_embbeding, com_embeddings)))


        adv_loss = 1 - torch.mean(torch.stack(com_loss_list))
        struct_loss = msssim(x, adv)

        if struct_loss < th:
            mu = mu * 0.6
        else:
            mu=mu*1.2
        if adv_loss < 1.0:
            mu = mu * 1.2

        total_loss = adv_loss * mu + beta * struct_loss

        grad = torch.autograd.grad(total_loss, adv,
                                   retain_graph=False, create_graph=False)[0]

        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad + momentum * decay
        momentum = grad

        grad = conv2d_same_padding(grad, gaussian_kernel, stride=1, padding=1, groups=3)

        amplification += alpha_beta * torch_staircase_sign(grad, 1.5625)
        cut_noise = clip_by_tensor(abs(amplification) - eps, 0.0, 10000.0) * torch.sign(amplification)
        projection = alpha * torch_staircase_sign(project_noise(cut_noise, stack_kern, kern_size), 1.5625)
        pert = (alpha_beta * torch_staircase_sign(grad, 1.5625) + 0.5 * projection) * 0.75
        adv = adv.detach() + pert

        delta = torch.clamp(adv - x, min=-eps, max=eps).detach()
        adv = clip_by_tensor(x + delta, 0, 1)

        # print("Current Step:",i+1,"Adv_loss:", adv_loss.cpu().item(),"  MSSSIM:", struct_loss.item(),)

    return adv


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../source_data/images')
    parser.add_argument('--output_dir', type=str, default='../result_data/images')
    parser.add_argument('--max_epsilon', type=float, default=8, help="Maximum size of adversarial perturbation")
    parser.add_argument('--threshold', type=float, default=0.96, help="The Min MSSIM threshold")
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--cuda', type=int, default=0, help="CUDA")
    parser.add_argument('--momentum', type=float, default=0.3, help="Momentum")
    parser.add_argument('--resize_chance', type=float, default=0.7, help="The probability of input shape")
    parser.add_argument('--resize_rate', type=float, default=np.array([1.25, 1.5, 1.75, 2]))
    parser.add_argument("--amplification", type=float, default=1.5, help="To amplifythe step size.")
    parser.add_argument('--model_list', type=list, default=['ir50_ms1m','IR_50_LFW_AdvTrain','IR_101','ir152', 'irse50', 'mobile_face', 'facenet'], help='models to ensemble')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device('cuda')

    models_info = set_model_info(args.model_list,device)
    max_epsilon=args.max_epsilon
    threshold=args.threshold
    dirlist=os.listdir(args.input_dir)
    dirlist.sort()
    for dir_name in  dirlist:
        name_lists=os.listdir(os.path.join(args.input_dir, dir_name))
        name_lists.sort()
        if not name_lists[0].endswith('jpg'):
            name_lists=name_lists[1:]
        for i,name in enumerate(name_lists):
            if i == 0:
                com_img_name = name_lists[1]
            else:
                com_img_name = name_lists[0]
            if '.jpg' not in name:
                continue
            if os.path.exists(os.path.join(args.output_dir,dir_name,name)):
                continue
            print('Deal with %s,Compare with %s' % (name, com_img_name))
            img = read_img_from_path(os.path.join( args.input_dir, dir_name), name, device)
            com_img = read_img_from_path(os.path.join( args.input_dir, dir_name), com_img_name, device)

            if not os.path.exists(os.path.join(args.output_dir, dir_name)):
                os.makedirs(os.path.join(args.output_dir, dir_name))

            cv2_save(img, os.path.join(args.output_dir, dir_name, name))
            png_name = os.path.join(args.output_dir, dir_name, name.replace('jpg','png'))
            jpg_name = os.path.join(args.output_dir, dir_name, name)
            os.rename(png_name, jpg_name)

            args.max_epsilon=max_epsilon
            args.threshold=threshold

            flage = compareIm(os.path.join(args.input_dir, dir_name, com_img_name),
                                    os.path.join(args.output_dir, dir_name, name))
            if flage == 0:
                adv_img = graph(img, com_img, args, models_info, name)
                cv2_save(adv_img, os.path.join(args.output_dir, dir_name, name))
                png_name = os.path.join(args.output_dir, dir_name, name.replace('jpg', 'png'))
                jpg_name = os.path.join(args.output_dir, dir_name, name)
                os.rename(png_name, jpg_name)
            while flage>65:
                adv_img = graph(img,com_img,args, models_info, name)
                args.max_epsilon=args.max_epsilon+1
                args.threshold=args.threshold-0.01
                cv2_save(adv_img, os.path.join(args.output_dir, dir_name, name))
                png_name = os.path.join(args.output_dir, dir_name, name.replace('jpg', 'png'))
                jpg_name = os.path.join(args.output_dir, dir_name, name)
                os.rename(png_name, jpg_name)
                flage = compareIm(os.path.join(args.input_dir, dir_name, com_img_name),
                                    os.path.join(args.output_dir, dir_name, name))
                if args.threshold<0.88:
                    args.threshold=0.88
                if args.max_epsilon>18:
                    break
            print(name,flage)


