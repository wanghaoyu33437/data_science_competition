
import os

import transformers
from transformers import BertModel,BertTokenizer

NEW_DIR = 'model_data/pretrain-model/macbert-base'
print('Transformers version',transformers.__version__)

def transformers_model_dowloader(pretrained_model_name_list = ['bert-base-uncased'], is_tf = True):

    for i, pretrained_model_name in enumerate(pretrained_model_name_list):
        print(i,'/',len(pretrained_model_name_list))
        print("Download model and tokenizer", pretrained_model_name)
        transformer_model = BertModel.from_pretrained(pretrained_model_name)
        transformer_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

        print("Save model and tokenizer", pretrained_model_name, 'in directory', NEW_DIR)
        transformer_model.save_pretrained(NEW_DIR)
        transformer_tokenizer.save_pretrained(NEW_DIR)

os.makedirs(NEW_DIR, exist_ok=True)
pretrained_model_name_list = ['hfl/chinese-macbert-base']
transformers_model_dowloader(pretrained_model_name_list, is_tf = False)
print("Download finish")