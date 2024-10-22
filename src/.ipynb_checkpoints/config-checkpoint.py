

cfg_unet_medical = {
    'model': 'unet_medical',
    'crop': [388 / 572, 388 / 572],
    'img_size': [572, 572],
    'lr': 0.0001,
    'epochs': 400,
    'repeat': 400,
    'distribute_epochs': 1600,
    'batchsize': 16,
    'cross_valid_ind': 1,
    'num_classes': 2,
    'num_channels': 1,

    'keep_checkpoint_max': 10,
    'weight_decay': 0.0005,
    'loss_scale': 1024.0,
    'FixedLossScaleManager': 1024.0,

    'resume': False,
    'resume_ckpt': './',
    'transfer_training': False,
    'filter_weight': ['outc.weight', 'outc.bias'],
    'eval_activate': 'Softmax',
    'eval_resize': False
}

cfg_unet_nested = {
    'model': 'unet_nested',
    'crop': None,
    'img_size': [576, 576],
    'lr': 0.0001,
    'epochs': 400,
    'repeat': 400,
    'distribute_epochs': 1600,
    'batchsize': 16,
    'cross_valid_ind': 1,
    'num_classes': 2,
    'num_channels': 1,

    'keep_checkpoint_max': 10,
    'weight_decay': 0.0005,
    'loss_scale': 1024.0,
    'FixedLossScaleManager': 1024.0,
    'use_bn': True,
    'use_ds': True,
    'use_deconv': True,

    'resume': False,
    'resume_ckpt': './',
    'transfer_training': False,
    'filter_weight': ['final1.weight', 'final2.weight', 'final3.weight', 'final4.weight'],
    'eval_activate': 'Softmax',
    'eval_resize': False
}

cfg_unet_nested_cell = {
    'model': 'unet_nested',
    'dataset': 'Cell_nuclei',
    'crop': None,
    'img_size': [96, 96],
    'lr': 3e-4,
    'epochs': 200,
    'repeat': 10,
    'distribute_epochs': 1600,
    'batchsize': 16,
    'cross_valid_ind': 1,
    'num_classes': 2,
    'num_channels': 3,

    'keep_checkpoint_max': 10,
    'weight_decay': 0.0005,
    'loss_scale': 1024.0,
    'FixedLossScaleManager': 1024.0,
    'use_bn': True,
    'use_ds': True,
    'use_deconv': True,

    'resume': False,
    'resume_ckpt': './',
    'transfer_training': False,
    'filter_weight': ['final1.weight', 'final2.weight', 'final3.weight', 'final4.weight'],
    'eval_activate': 'Softmax',
    'eval_resize': False
}

cfg_unet_simple = {
    'model': 'unet_simple',
    'crop': None,
    'img_size': [576, 576],  # 输入图片size
    'lr': 0.0001,
    'epochs': 400,
    'repeat': 400,
    'distribute_epochs': 1600,
    'batchsize': 16,
    'cross_valid_ind': 1,
    'num_classes': 2,   # 2类，前景背景
    'num_channels': 1,  # 

    'keep_checkpoint_max': 10,
    'weight_decay': 0.0005,
    'loss_scale': 1024.0,
    'FixedLossScaleManager': 1024.0,

    'resume': False,
    'resume_ckpt': './',
    'transfer_training': False,
    'filter_weight': ["final.weight"],
    'eval_activate': 'Softmax',
    'eval_resize': False
}

cfg_unet_simple_coco = {
    'model': 'unet_simple',
    'dataset': 'COCO',
    'split': 0.8,
    # 填写该处的图片输入尺寸，如'img_size': [512, 512],  w/h
    #-----------****************
    'img_size': [],
    
    
    #-----------*****************
    'lr': 1e-4,
    #填写该处训练步数：
    #-----------****************
    'epochs':,
    
    #-----------*****************
    
    
    #填写该处训练参数：
    #-----------****************
    'repeat': ,
    'distribute_epochs': ,
    'cross_valid_ind': ,
    'batchsize': ,
    'num_channels': ,
    
    #-----------*****************
    

    'keep_checkpoint_max': 10,
    'weight_decay': 0.0005,
    'loss_scale': 1024.0,
    'FixedLossScaleManager': 1024.0,

    'resume': False,
    'resume_ckpt': './',
    'transfer_training': False,
    'filter_weight': ["final.weight"],
    'eval_activate': 'Softmax',
    'eval_resize': False,
    
    #填写类别数目、类别名称和coco文件夹路径：
    
    #-----------****************
    'num_classes': ,
    'coco_classes': (  , 'colloid'),
    'coco_dir': ' ',
    
    #-----------*****************
    'anno_json': './raw_data/annotations/instances_annotations.json',
    


    'val_anno_json': '/data/coco2017/annotations/instances_val2017.json',
    'val_coco_dir': '/data/coco2017/val2017'
}

cfg_unet = cfg_unet_simple_coco
if not ('dataset' in cfg_unet and cfg_unet['dataset'] == 'Cell_nuclei') and cfg_unet['eval_resize']:
    print("ISBI dataset not support resize to original image size when in evaluation.")
    cfg_unet['eval_resize'] = False
