import logging
import os
import time
import torch
# os.environ['CUDA_VISIBLE_DEVICES']= '0,1,2,3'
from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal,MyBert
import warnings
warnings.filterwarnings('ignore')

from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from tqdm import tqdm
from adv_train import FGM,PGD,EMA,ExponentialMovingAverage
from data_helper import MultiModalDataset
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

import torch.utils.data
import numpy as np
def train_and_validate(args):
    # 1. load data
    # for f,(train_dataloader, val_dataloader) in  enumerate(create_dataloaders(args)):
    #     os.makedirs(args.savedmodel_path+'/'+str(f), exist_ok=True)
    train_dataloader,val_dataloader = create_dataloaders(args)
    unlabeled_dataset = MultiModalDataset(args, args.unlabeled_annotation, args.unlabeled_zip_feats, test_mode=True)
    sampler = torch.utils.data.SequentialSampler(unlabeled_dataset)
    # sampler = torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(len(unlabeled_dataset))))
    # sampler = torch.utils.data.Subset()
    unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_dataset,
                            batch_size=args.batch_size,
                            sampler=sampler,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)


    os.makedirs(args.savedmodel_path, exist_ok=True)
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
    ema = EMA(model,0.99)
    ema.register()

    for epoch in tqdm(range(args.max_epochs)):
        # model.train()
        alpha = 1 * (epoch+1) / args.max_epochs
        iters=len(train_dataloader)
        for i,(batch_ul,batch) in tqdm(enumerate(zip(unlabeled_dataloader,train_dataloader))):
            for k, v in batch_ul.items():
                batch_ul[k] = v.to(args.device)
            for k, v in batch.items():
                batch[k] = v.to(args.device)

            model.eval()
            with torch.no_grad():
                pred = model(batch_ul, inference=True)
                pred_label_id = torch.argmax(pred, dim=1)
                batch_ul['label'] = pred_label_id.unsqueeze(1)

            model.train()
            optimizer.zero_grad()
            with autocast():
                loss, _, _, _ = model(batch_ul)
                loss = loss.mean()
            scaler.scale(alpha*loss).backward()

            with autocast():
                loss, accuracy, pred_id, label = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
            scaler.scale(loss).backward()

            with autocast():
                fgm.attack(epsilon=1., emb_name='word_embeddings')
                loss_adv, accuracy_adv, _, _ = model(batch)
                loss_adv = loss_adv.mean()
                accuracy_adv = accuracy_adv.mean()
            scaler.scale(loss_adv).backward()
            fgm.restore(emb_name='word_embeddings')

            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            ema.update()
            scheduler.step()
            step += 1
            if step % args.print_steps == 0:
                lr = optimizer.param_groups[0]['lr']
                # pred_id =pred_id.cpu().numpy()
                # label = label.cpu().numpy()
                # train_results = evaluate(pred_id,label)
                # train_results = {k: round(v, 4) for k, v in train_results.items()}
                # time_per_step = (time.time() - start_time) / max(1, step)
                # remaining_time = time_per_step * (num_total_steps - step)
                # remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(
                    f"Fold  Epoch {epoch} step {step} loss {loss:.3f}, accuracy {accuracy:.3f},loss_adv {loss_adv:.3f}, accuracy_adv {accuracy_adv:.3f} , lr {lr}")
                # logging.info(train_results)
        # 4. validation
        loss, results = validate(model, val_dataloader,args)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Fold  Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
        ema.apply_shadow()
        loss, results = validate(model, val_dataloader,args)
        ema.restore()
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Fold  Model EMA Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
        # scheduler.step()
        # 5. save checkpoint
        mean_f1 = results['mean_f1']
        # if mean_f1 > best_score:
        #     best_score = mean_f1
        state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
        ema.apply_shadow()
        ema_state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
        ema.restore()
        torch.save({'epoch': epoch, 'model_state_dict': state_dict,'model_ema_state_dict':ema_state_dict, 'mean_f1': mean_f1},
                       f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')

import time
if __name__ == '__main__':
    args = parse_args()
    setup_device(args)
    setup_seed(args)
    os.makedirs(args.savedmodel_path, exist_ok=True)
    setup_logging(args.savedmodel_path)

    logging.info("Training/evaluation parameters: %s", args)
    shutil.copy('main_semi_advTrain.py',args.savedmodel_path+'/main_semi_advTrain.py')
    shutil.copy('config.py',args.savedmodel_path+'/config.py')
    shutil.copy('model.py',args.savedmodel_path+'/model.py')
    train_and_validate(args)
    # main()
