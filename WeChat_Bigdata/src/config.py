import argparse
import time

def parse_args():
    date = time.asctime().split()
    date_save_path_params = '%s_%s_1_MyBert_advTrainFGM'%(date[1],date[2])
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=7777, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
    parser.add_argument('--KFold',type = int,default=5,help='KFold')
    parser.add_argument('--fold',type=list,default=[4])

    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='data/annotations/labeled.json')
    parser.add_argument('--test_annotation', type=str, default='data/annotations/test_a.json')
    parser.add_argument('--unlabeled_annotation', type=str, default='data/annotations/unlabeled_10w.json')
    parser.add_argument('--train_zip_feats', type=str, default='data/zip_feats/labeled.zip')
    parser.add_argument('--test_zip_feats', type=str, default='data/zip_feats/test_a.zip')
    parser.add_argument('--unlabeled_zip_feats', type=str, default='data/zip_feats/unlabeled.zip')

    parser.add_argument('--val_ratio', default=0.1, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=32, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=256, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=256, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='model_data/v1/%s'%date_save_path_params)
    parser.add_argument('--ckpt_file', type=str, default='model_data/v1/Jun_12_1_MyBert_advTrainFGM/model_epoch_5_mean_f1_0.6655.bin')
    parser.add_argument('--test_output_csv', type=str, default='model_data/v1/Jun_12_1_MyBert_advTrainFGM/result.csv')
    parser.add_argument('--test_output_npy', type=str,default='model_data/v1/Jun_12_1_MyBert_advTrainFGM/result_pred.npy')
    parser.add_argument('--best_score', default=0.5, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=20, help='How many epochs')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=50, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=2, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0, type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--model_ema_steps', default=32, type=int)
    parser.add_argument('--model_ema_decay', default=0.98, type=float)

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='hfl/chinese-macbert-base')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    # parser.add_argument('--bert_seq_length', type=int, default=50)
    parser.add_argument('--bert_seq_length', type=int, default=100)
    parser.add_argument('--bert_learning_rate', type=float, default=3e-5)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument('--bert_output_size', type=int, default=768)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=8)
    parser.add_argument('--vlad_hidden_size', type=int, default=768, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=1024, help="linear size before final linear")
    # parser.add_argument('--fc_size', type=int, default=1024, help="linear size before final linear")

    return parser.parse_args()
