import logging
import os
import time
import torch
os.environ['CUDA_VISIBLE_DEVICES']= '0,1'
from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal,MyBert
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from tqdm import tqdm

import shutil

scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast

def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
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
        os.makedirs(args.savedmodel_path+'/'+str(f), exist_ok=True)
        args.max_steps = len(train_dataloader) * args.max_epochs
        args.warmup_steps = len(train_dataloader)*int(args.max_epochs/10)
        # 2. build model and optimizers
        # model = MultiModal(args)
        model = MyBert.from_pretrained(args.bert_dir,cache_dir=args.bert_cache,args=args)
        optimizer, scheduler = build_optimizer(args, model)
        if args.device == 'cuda':
            model = torch.nn.parallel.DataParallel(model.to(args.device))

        # 3. training
        step = 0
        best_score = args.best_score
        start_time = time.time()

        for epoch in tqdm(range(args.max_epochs)):
            model.train()

            iters=len(train_dataloader)
            for i,batch in enumerate(tqdm(train_dataloader)):
                for k, v in batch.items():
                    batch[k] = v.to(args.device)
                optimizer.zero_grad()
                with autocast():
                    loss, accuracy, pred_id, label = model(batch)
                    # model(batch['title_input'],batch['title_mask'],batch['frame_input'],batch['frame_mask'])
                    loss = loss.mean()
                    accuracy = accuracy.mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # loss.backward()
                # optimizer.step()
                # scheduler.step(epoch+i/iters)
                scheduler.step()
                step += 1
                if step % args.print_steps == 0:
                    lr = optimizer.param_groups[0]['lr']
                    # time_per_step = (time.time() - start_time) / max(1, step)
                    # remaining_time = time_per_step * (num_total_steps - step)
                    # remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    logging.info(f"fold {f} Epoch {epoch} step {step} loss {loss:.3f}, accuracy {accuracy:.3f} , lr {lr}")

            # 4. validation
            loss, results = validate(model, val_dataloader)
            results = {k: round(v, 4) for k, v in results.items()}
            logging.info(f"fold {f} Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
            # scheduler.step()
            # 5. save checkpoint
            mean_f1 = results['mean_f1']
            if mean_f1 > best_score:
                best_score = mean_f1
                state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()

                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                           f'{args.savedmodel_path}/{f}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')
        del model
        del train_dataloader
        del val_dataloader
        torch.cuda.empty_cache()


import time
if __name__ == '__main__':
    args = parse_args()
    setup_device(args)
    setup_seed(args)
    os.makedirs(args.savedmodel_path, exist_ok=True)
    setup_logging(args.savedmodel_path)

    logging.info("Training/evaluation parameters: %s", args)
    shutil.copy('main.py',args.savedmodel_path+'/main.py')
    shutil.copy('config.py',args.savedmodel_path+'/config.py')
    shutil.copy('model.py',args.savedmodel_path+'/model.py')
    train_and_validate(args)
    # main()
