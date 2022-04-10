params_config={
    'bce':0.7,
    'dice':0.3,
    'focal':1,
    'optimizer_name': 'AdamW',
    'optimizer_hyperparameters': {
        'lr': 3e-4,  # 0.001,
        # 'momentum': 0.9,
        'weight_decay': 5e-4
    },
    'scheduler_name': 'CosineAnnealingWarmRestarts',
    'scheduler_hyperparameters': {
        'T_0': 3,
        'T_mult': 2,
        'eta_min': 1e-6,
    }
}
model_configs=[
    {
        'name':'model_1', # 2752
        'data_path':'../user_data/flist/Mar14_train768_val500_s3_500.npy',
        'train_size':768,
        'batch_size':16,
        'epoch':21,# 21
        'transform':{
            'Copy_move':0.5,
            'random_paint':0.3,
        }
     },
    {
        'name': 'model_2', # 2757
        'data_path': '../user_data/flist/Mar14_train768_val500_s3_500.npy',
        'train_size': 768,
        'batch_size': 16,
        'epoch': 37, #37
        'transform': {
            'Copy_move': 0,
            'random_paint': 0,
        },
    },
    {
        'name': 'model_3', # 2766
        'data_path': '../user_data/flist/Mar16_train4000_s3_2000.npy',
        'train_size': 768,
        'batch_size': 16,
        'epoch': 92, #92
        'transform': {
            'Copy_move': 0.1,
            'random_paint': 0,
        }
    },
    {
        'name': 'model_4',  # 2764
        'data_path': '../user_data/flist/Mar16_train4000_s3_1000_book579.npy',
        'train_size': 896,
        'batch_size': 12,
        'epoch': 92, # 02
        'transform': {
            'Copy_move': 0.1,
            'random_paint': 0.1,
        }
    },
    {
        'name': 'model_5',  # 2756
        'data_path': '../user_data/flist/Mar19_train4000_s3_2000_book1446.npy',
        'train_size': 768,
        'batch_size': 16,
        'epoch': 91, #91
        'transform': {
            'Copy_move': 0.2,
            'random_paint': 0.,
        }
    }



]