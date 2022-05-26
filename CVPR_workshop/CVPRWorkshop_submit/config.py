
args_resnet50 = {
    'name': 'resnet50',
    'epochs': 300,  # {2:[5,13,29,61],3:[8,20,44,92]}
    'model_hyperparameters': {
        'pretrained': False,
        'num_classes': 1,
        'drop_path_rate': 0.1,
    },
    # 'update_freq':1,
    # 'drop_path':0.1,
    'optimizer_name': 'AdamW',  # 'AdamW',#'SGD
    'optimizer_hyperparameters': {
        'lr': 1e-3,  # 0.001,
        # 'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'min_lr': 5e-6,
    'warmup_epochs': 20,  # epochs to warmup LR, if scheduler supports
    'warmup_steps': 1,  # num of steps to warmup LR, will overload warmup_epochs if set > 0
    'warmup_lr': 5e-7,
    'decay_rate': 0.1,
    'decay_epochs': 30,
    'resume': False,
    'resume_path': '/',
    'scheduler_name': 'CosineLRScheduler',
    'batch_size': 128,
}