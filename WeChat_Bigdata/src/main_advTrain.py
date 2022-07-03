import logging
import os
import time
import torch
os.environ['CUDA_VISIBLE_DEVICES']='0'
from config import parse_args
from data_helper import create_dataloaders

from model import MultiModal,MyBert
import warnings
warnings.filterwarnings('ignore')

from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from tqdm import tqdm
from adv_train import FGM,PGD,EMA,ExponentialMovingAverage

import shutil


scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast


def validate(model, val_dataloader, args):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            for k, v in batch.items():
                batch[k] = v.to(args.device)
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)


    model.train()
    return loss, results


def train_and_validate(args):
    # 1. load data
    for f,(train_dataloader, val_dataloader) in  enumerate(create_dataloaders(args)):
        if f not in args.fold :
            continue
        os.makedirs(args.savedmodel_path+'/'+str(f), exist_ok=True)
        # train_dataloader,val_dataloader = create_dataloaders(args)
        # os.makedirs(args.savedmodel_path, exist_ok=True)
        args.max_steps = len(train_dataloader) * args.max_epochs
        args.warmup_steps = len(train_dataloader)*max(1,int(args.max_epochs/10))
        logging.info(f'max_steps {args.max_steps} warmup_steps {args.warmup_steps}')
        #
        # 2. build model and optimizers
        # model = MultiModal(args)
        model = MyBert.from_pretrained(args.bert_dir,cache_dir=args.bert_cache,args=args)
        optimizer, scheduler = build_optimizer(args, model)

        if args.device == 'cuda':
            model = torch.nn.parallel.DataParallel(model.to(args.device))
        # model_ema = ExponentialMovingAverage(model, device=args.device, decay=args.model_ema_decay)
        # 5. training
        step = 0
        best_score = args.best_score
        start_time = time.time()
        fgm = FGM(model)
        # pgd = PGD(model)
        # steps_for_at = 3
        ema = EMA(model, args.model_ema_decay)
        ema_state_dict = None
        for epoch in tqdm(range(args.max_epochs)):
            model.train()

            iters = len(train_dataloader)
            for i, batch in enumerate(tqdm(train_dataloader)):
                for k, v in batch.items():
                    batch[k] = v.to(args.device)
                optimizer.zero_grad()

                with autocast():
                    loss_ce, accuracy, pred_id, label = model(batch)
                    loss_ce = loss_ce.mean()
                    pred_id = pred_id.cpu().numpy()
                    label = label.cpu().numpy()
                    results = evaluate(pred_id, label)
                    f1_mean = results['mean_f1']
                    loss = loss_ce - f1_mean
                    accuracy = accuracy.mean()
                # loss.backward()
                scaler.scale(loss).backward()

                # 保存正常的gradient
                # pgd.backup_grad()
                # for t in range(steps_for_at):
                #     # 在embedding上添加对抗扰动, first attack时备份param.data
                #     pgd.attack(is_first_attack=(t == 0))
                #     # 中间过程，梯度清零
                #     if t != steps_for_at - 1:
                #         optimizer.zero_grad()
                #     # 最后一步，恢复正常的grad
                #     else:
                #         pgd.restore_grad()
                #     # embedding参数被修改，此时，输入序列得到的embedding表征不一样
                #     loss_adv, accuracy_adv, _, _ = model(batch)
                #     loss_adv = loss_adv.mean()
                #     accuracy_adv = accuracy_adv.mean()
                #     # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                #     scaler.scale(loss_adv).backward()
                # pgd.restore()
                # if step >= args.warmup_steps:
                with autocast():
                    fgm.attack(epsilon=1., emb_name='word_embeddings')
                    loss_ce_adv, accuracy_adv, pred_id_adv, label = model(batch)
                    loss_ce_adv = loss_ce_adv.mean()
                    accuracy_adv = accuracy_adv.mean()
                    pred_id_adv = pred_id_adv.cpu().numpy()
                    label = label.cpu().numpy()
                    results = evaluate(pred_id_adv, label)
                    f1_mean_adv = results['mean_f1']

                loss_adv = loss_ce_adv - f1_mean_adv
                # loss_adv.backward()
                scaler.scale(loss_adv).backward()
                fgm.restore(emb_name='word_embeddings')
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                if step == args.warmup_steps:
                    ema.register()

                if ema and i % args.model_ema_steps == 0:
                    if step >= args.warmup_steps:
                        ema.update()

                # loss.backward()
                # optimizer.step()
                scheduler.step()
                step += 1
                if step % args.print_steps == 0:
                    lr = optimizer.param_groups[0]['lr']
                    logging.info(
                        f"Fold {f} Epoch {epoch} step {step} loss {loss:.3f}, accuracy {accuracy:.3f} f1_mean {f1_mean} lr {lr}")
                    if step >args.warmup_steps:
                        logging.info(
                            f'Fold {f}  Epoch {epoch} step {step} loss_adv{loss_adv: .3f} accuracy_adv {accuracy_adv: .3f} f1_mean_adv {f1_mean_adv}')
                    # logging.info(train_results)
            # 4. validation
            loss, results = validate(model, val_dataloader, args)
            results = {k: round(v, 4) for k, v in results.items()}
            mean_f1 = results['mean_f1']
            logging.info(f"Fold {f}  Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
            mean_f1_mea = 0
            if step > args.warmup_steps:
                ema.apply_shadow()
                loss, results = validate(model, val_dataloader, args)
                ema.restore()
                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Fold {f}  Model EMA Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
                mean_f1_mea = results['mean_f1']
            # scheduler.step()
            # 5. save checkpoint

            # if mean_f1 > best_score:
            #     best_score = mean_f1
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            if step > args.warmup_steps:
                ema.apply_shadow()
                ema_state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                ema.restore()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'model_ema_state_dict': ema_state_dict,
                        'mean_f1': mean_f1},
                       f'{args.savedmodel_path}/{f}/model_epoch_{epoch}_mean_f1_{mean_f1}_ema{mean_f1_mea}.bin')


import time
if __name__ == '__main__':
    args = parse_args()
    setup_device(args)
    setup_seed(args)
    os.makedirs(args.savedmodel_path, exist_ok=True)
    setup_logging(args.savedmodel_path)

    logging.info("Training/evaluation parameters: %s", args)
    shutil.copy('main_advTrain.py',args.savedmodel_path+'/main_advTrain.py')
    shutil.copy('config.py',args.savedmodel_path+'/config.py')
    shutil.copy('model.py',args.savedmodel_path+'/model.py')
    train_and_validate(args)
    # main()
