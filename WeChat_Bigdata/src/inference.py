import torch
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal,MyBert
from tqdm import tqdm
import numpy as np
def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. load model
    # model = MultiModal(args)
    model = MyBert.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, args=args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.load_state_dict(checkpoint['model_ema_state_dict'])
    model.eval()

    # 3. inference
    predictions = []
    pred_label_ids=[]
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pred = model(batch, inference=True)
            pred_label_id = torch.argmax(pred,dim=1)
            predictions.extend(pred.cpu().numpy())
            pred_label_ids.extend(pred_label_id.cpu().numpy())
    # 4. dump results
    id_pred=[]
    with open(args.test_output_csv, 'w') as f:
        for pred,pred_label_id,ann in zip(predictions,pred_label_ids, dataset.anns):
            video_id = ann['id']
            id_pred.append([video_id,pred])
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')
    np.save(args.test_output_npy,id_pred)

if __name__ == '__main__':
    inference()
