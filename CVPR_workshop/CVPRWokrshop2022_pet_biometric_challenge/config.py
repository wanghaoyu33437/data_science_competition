args_resnet50 = {
    'name':'resnet50',
    'epochs': 185,#{2:[5,13,29,61],3:[8,20,44,92]}
    'optimizer_name': 'AdamW',#'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 0.001,#0.001,
        # 'momentum': 0.9,
        # 'nesterov':True,
        'weight_decay': 5e-4
    },
    'scheduler_name': 'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters': {
        'T_0':3,
        'T_mult':2,
        'eta_min':1e-5,
    },
    'resume': False,
    'resume_path': '',
    'batch_size': 256,
}
args_resnet50_afterPreTrain = {
    'name':'resnet50',
    'epochs': 95,#{2:[5,13,29,61],3:[8,20,44,92]}
    'optimizer_name': 'AdamW',#'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 0.001,#0.001,
        # 'momentum': 0.9,
        # 'nesterov':True,
        'weight_decay': 5e-4
    },
    'scheduler_name': 'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters': {
        'T_0':3,
        'T_mult':2,
        'eta_min':1e-5,
    },
    'resume': True,
    'resume_path': 'Phase2_model_data/Apr_13_Resnet50_Train_clean_all_adv4_22k_advCor_1of2_Patch_1of2_size224/resnet50_epoch92_T99.4773_V99.6962.pth',
    'batch_size': 64,
}
args_resnext101 = {
    'name':'resnext101_32x4d',
    'epochs': 185,#{2:[5,13,29,61],3:[8,20,44,92]}
    'optimizer_name': 'AdamW',#'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 0.001,#0.001,
        # 'momentum': 0.9,
        'weight_decay': 5e-4
    },
    'scheduler_name': 'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters': {
        'T_0':3,
        'T_mult':2,
        'eta_min':1e-5,
    },
    'resume': False,
    'resume_path': '//',

    'batch_size': 256,
}
args_seresnext101_32x8d = {
    'name':'seresnext101_32x8d',
    'epochs': 185,#{2:[5,13,29,61],3:[8,20,44,92]}
    'optimizer_name': 'AdamW',#'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 0.001,#0.001,
        # 'momentum': 0.9,
        # 'nesterov':True,
        'weight_decay': 5e-4
    },
    'scheduler_name': 'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters': {
        'T_0':3,
        'T_mult':2,
        'eta_min':1e-5,
    },
    'resume':False,
    'resume_path':'Phase1_model_data/Mar_30_Seresnext101_32x8d_all_PGD_030_MTI_all_AdvPatch_corruption_preTrain/seresnext101_32x8d_epoch23_T69.2275.pth',
    'batch_size': 48,
}
args_seresnext101_32x8d_Pre_train = {
    'name':'seresnext101_32x8d',
    'epochs': 30,#{2:[5,13,29,61],3:[8,20,44,92]}
    'optimizer_name': 'SGD',#'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 0.01,#0.001,
        'momentum': 0.9,
        'nesterov':True,
        'weight_decay': 5e-4
    },
    'scheduler_name': 'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters': {
        'T_0':10,
        'T_mult':2,
        'eta_min':1e-5,
    },
    'batch_size': 64,
}
args_desnet121 = {
    'name':'densenet121',
    'epochs': 185,#{2:[5,13,29,61],3:[8,20,44,92]}
    'optimizer_name': 'AdamW',#'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 0.001,#0.001,
        # 'momentum': 0.9,
        'weight_decay': 5e-4
    },
    'resume': False,
    'resume_path': '//',
    'scheduler_name': 'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters': {
        'T_0':3,
        'T_mult':2,
        'eta_min':1e-5,
    },
    'batch_size': 256,
}

args_mobilenetv3_large_100 = {
    'name':'mobilenetv3_large_100',
    'epochs': 185,#{2:[5,13,29,61],3:[8,20,44,92]}
    'optimizer_name': 'AdamW',#'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 0.001,#0.001,
        # 'momentum': 0.9,
        'weight_decay': 5e-4
    },
    'scheduler_name': 'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters': {
        'T_0':3,
        'T_mult':2,
        'eta_min':1e-5,
    },
    'batch_size': 128,
}
args_convnext_tiny_hnf = {
    'name':'convnext_tiny_hnf',
    'epochs': 300,#{2:[5,13,29,61],3:[8,20,44,92]}
    'model_hyperparameters':{
        'pretrained' :True,
        'num_classes':6000,
         'drop_path_rate':0.1,
    },
    # 'update_freq':1,
    # 'drop_path':0.1,
    'optimizer_name': 'AdamW',#'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 4e-3,#0.001,
        # 'momentum': 0.9,
        'weight_decay': 5e-2
    },
    'min_lr':1e-6,
    'warmup_epochs':20, # epochs to warmup LR, if scheduler supports
    'warmup_steps':1,# num of steps to warmup LR, will overload warmup_epochs if set > 0
    'warmup_lr' : 5e-7,
    'decay_rate': 0.1,
    'decay_epochs':30,
    'resume': False,
    'resume_path': '//',
    'scheduler_name': 'CosineLRScheduler',
    'batch_size': 256,
}

args_convnext_base = {
    'name':'convnext_base',
    'epochs': 300,#{2:[5,13,29,61],3:[8,20,44,92]}
    'model_hyperparameters':{
        'pretrained' :False,
        'num_classes':6000,
         'drop_path_rate':0.1,
    },
    # 'update_freq':1,
    # 'drop_path':0.1,
    'optimizer_name': 'AdamW',#'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 4e-3,#0.001,
        # 'momentum': 0.9,
        'weight_decay': 5e-2
    },
    'min_lr':1e-6,
    'warmup_epochs':20, # epochs to warmup LR, if scheduler supports
    'warmup_steps':1,# num of steps to warmup LR, will overload warmup_epochs if set > 0
    'warmup_lr' : 5e-7,
    'decay_rate': 0.1,
    'decay_epochs':30,
    'resume': False,
    'resume_path': '//',
    'scheduler_name': 'CosineLRScheduler',
    'batch_size': 256,
}


args_swin_tiny_patch4_window7_224= {
    'name':'swin_tiny_patch4_window7_224',
    'epochs': 300,#{2:[5,13,29,61],3:[8,20,44,92]}
    'optimizer_name': 'AdamW',#'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 5e-4,#0.001,
        # 'momentum': 0.9,
        'weight_decay': 5e-2,
    },
    'min_lr': 5e-6,
    'warmup_epochs': 20,
    'warmup_lr' : 5e-7,
    'decay_epochs':30,
    'decay_rate':0.1,
    # timm scheduler
    'resume': False,
    'resume_path': '//',
    'scheduler_name': 'CosineLRScheduler',
    'batch_size': 64,
}
args_swin_s3_tiny_224 = {
    'name':'swin_s3_tiny_224',
    'epochs': 300,  # {2:[5,13,29,61],3:[8,20,44,92]}
    'model_hyperparameters': {
        'pretrained': False,
        'num_classes': 6000,
        'drop_path_rate': 0.1,
    },
    # 'update_freq':1,
    # 'drop_path':0.1,
    'optimizer_name': 'AdamW',  # 'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 5e-4,  # 0.001,
        # 'momentum': 0.9,
        'weight_decay': 5e-2
    },
    'min_lr': 5e-6,
    'warmup_epochs': 20,  # epochs to warmup LR, if scheduler supports
    'warmup_steps': 1,  # num of steps to warmup LR, will overload warmup_epochs if set > 0
    'warmup_lr': 5e-7,
    'decay_rate': 0.1,
    'decay_epochs': 30,
    'resume': False,
    'resume_path': '//',
    'scheduler_name': 'CosineLRScheduler',
    'batch_size': 128,
}