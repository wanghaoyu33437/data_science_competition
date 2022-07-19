import torch
from utils import load_model

model  =load_model('resnet50',{'pretrained': False,'num_classes': 4})

model.load_state_dict(torch.load('model_data/Jun_27_Resnet50/resnet50_epoch54_Tacc_9975_Vacc_6781.pth',map_location='cpu'))
print('Load successfully ')
torch.save(model.state_dict(),'model_data/Jun_27_Resnet50/model.pth',_use_new_zipfile_serialization=False)
