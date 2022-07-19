
args_resnet50 = {
    'name': 'resnet50',
    'epochs': 100,  # {2:[5,13,29,61],3:[8,20,44,92]}
    'model_hyperparameters': {
        'pretrained': True,
        'num_classes': 4,
        'drop_path_rate': 0.1,
    },
    # 'update_freq':1,
    # 'drop_path':0.1,
    'optimizer_name': 'AdamW',  # 'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 1e-3,  # 0.001,
        # 'momentum': 0.9,
        'weight_decay': 1e-2
    },
    'min_lr': 5e-6,
    'warmup_epochs': 5,  # epochs to warmup LR, if scheduler supports
    'warmup_steps': 1,  # num of steps to warmup LR, will overload warmup_epochs if set > 0
    'warmup_lr': 5e-7,
    'decay_rate': 0.1,
    'decay_epochs': 30,
    'resume': False,
    'resume_path': '/',
    'scheduler_name': 'CosineLRScheduler',
    'batch_size': 128,
}


args_convnext_base = {
    'name': 'convnext_base',
    'epochs': 100,  # {2:[5,13,29,61],3:[8,20,44,92]}
    'model_hyperparameters': {
        'pretrained': True,
        'num_classes': 4,
        'drop_path_rate': 0.2,
    },
    # 'update_freq':1,
    # 'drop_path':0.1,
    'optimizer_name': 'AdamW',  # 'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 5e-4,  # 0.001,
        # 'momentum': 0.9,
        'weight_decay': 1e-2
    },
    'min_lr': 5e-6,
    'warmup_epochs': 5,  # epochs to warmup LR, if scheduler supports
    'warmup_steps': 1,  # num of steps to warmup LR, will overload warmup_epochs if set > 0
    'warmup_lr': 5e-7,
    'decay_rate': 0.1,
    'decay_epochs': 30,
    'resume': False,
    'resume_path': '/',
    'scheduler_name': 'CosineLRScheduler',
    # 'scheduler_name':'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters':{
        'T_0':3,
        'T_mult':2,
        'eta_min':1e-6,

    },
    'mixup':False,
    'LS':0.1,
    'ema_decay':0.99,
    'ema_step':1,
    'batch_size': 32,
    'input_size':224,
    'adv_train':True,
    'attack_method':"FGSM",
    "attack_method_params":{
        "eps":16/255,
        # "alpha":1/255,
        # "steps":20,
    }
}


args_convnext_tiny = {
    'name': 'convnext_tiny',
    'epochs': 50,  # {2:[5,13,29,61],3:[8,20,44,92]}
    'model_hyperparameters': {
        'pretrained': True,
        'num_classes': 4,
        'drop_path_rate': 0.2,
    },
    # 'update_freq':1,
    # 'drop_path':0.1,
    'optimizer_name': 'AdamW',  # 'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 5e-4,  # 0.001,
        # 'momentum': 0.9,
        # 'nesterov':True,
        'weight_decay': 5e-2
    },
    'min_lr': 1e-6,
    'warmup_epochs': 5,  # epochs to warmup LR, if scheduler supports
    'warmup_steps': 1,  # num of steps to warmup LR, will overload warmup_epochs if set > 0
    'warmup_lr': 5e-7,
    'decay_rate': 0.1,
    'decay_epochs': 30,
    'resume': False,
    'resume_path': '/',
    'scheduler_name': 'CosineLRScheduler',
    # 'scheduler_name':'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters':{
        'T_0':3,
        'T_mult':2,
        'eta_min':1e-6,
    },
    'mixup':1.0,
    'LS':0.1,
    "RDrop":0.3,
    'ema_decay':0.9999,
    'ema_step':1,
    'batch_size': 32,
    'input_size':384,
    'resize': 400,
    'adv_train':True,
    'attack_method':"FGSM",
    "attack_method_params":{
        "eps":20/255,
        # "alpha":1/255,
        # "steps":20,
    }
}




args_efficientnet = {
    'name': 'tf_efficientnet_b5',
    'epochs': 50,  # {2:[5,13,29,61],3:[8,20,44,92]}
    'model_hyperparameters': {
        'pretrained': True,
        'num_classes': 4,
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
    'warmup_epochs': 5,  # epochs to warmup LR, if scheduler supports
    'warmup_steps': 1,  # num of steps to warmup LR, will overload warmup_epochs if set > 0
    'warmup_lr': 5e-7,
    'decay_rate': 0.1,
    'decay_epochs': 30,
    'resume': False,
    'resume_path': '/',
    'scheduler_name': 'CosineLRScheduler',
    # 'scheduler_name':'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters':{
        'T_0':3,
        'T_mult':2,
        'eta_min':1e-6,

    },
    'mixup':0.7,
    'LS':0.0,
    'ema_decay':0.999,
    'ema_step':1,
    'batch_size': 32,
    'input_size':456,
    'adv_train':True,
    'attack_method':"FGSM",
    "attack_method_params":{
        "eps":20/255,
        # "alpha":1/255,
        # "steps":20,
    }
}
args_swin_tiny_patch4_window7_224 = {
    'name': 'swin_tiny_patch4_window7_224',
    'epochs': 50,  # {2:[5,13,29,61],3:[8,20,44,92]}
    'model_hyperparameters': {
        'pretrained': True,
        'num_classes': 4,
        'drop_path_rate': 0.2,
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
    'warmup_epochs': 5,  # epochs to warmup LR, if scheduler supports
    'warmup_steps': 1,  # num of steps to warmup LR, will overload warmup_epochs if set > 0
    'warmup_lr': 5e-7,
    'decay_rate': 0.1,
    'decay_epochs': 30,
    'resume': False,
    'resume_path': '/',
    'scheduler_name': 'CosineLRScheduler',
    # 'scheduler_name':'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters':{
        'T_0':3,
        'T_mult':2,
        'eta_min':1e-6,
    },
    'mixup':True,
    'LS':0.1,
    'ema_decay':0.999,
    'ema_step':1,
    'batch_size': 16,
    'input_size': 224,
    'adv_train':True,
    'attack_method':"FGSM",
    "attack_method_params":{
        "eps":16/255,
        # "alpha":1/255,
        # "steps":20,
    }
}

args_vit = {
    'name': 'vit_tiny_patch16_384',
    'epochs': 50,  # {2:[5,13,29,61],3:[8,20,44,92]}
    'model_hyperparameters': {
        'pretrained': True,
        'num_classes': 4,
        'drop_path_rate': 0.2,
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
    'warmup_epochs': 5,  # epochs to warmup LR, if scheduler supports
    'warmup_steps': 1,  # num of steps to warmup LR, will overload warmup_epochs if set > 0
    'warmup_lr': 5e-7,
    'decay_rate': 0.1,
    'decay_epochs': 30,
    'resume': False,
    'resume_path': '/',
    'scheduler_name': 'CosineLRScheduler',
    # 'scheduler_name':'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters':{
        'T_0':3,
        'T_mult':2,
        'eta_min':1e-6,
    },
    'mixup':1.0,
    'LS':0.1,
    'ema_decay':0.999,
    'ema_step':1,
    'batch_size': 32,
    'input_size':384,
    'resize': 400,
    'adv_train':True,
    'attack_method':"FGSM",
    "attack_method_params":{
        "eps":20/255,
        # "alpha":1/255,
        # "steps":20,
    }
}