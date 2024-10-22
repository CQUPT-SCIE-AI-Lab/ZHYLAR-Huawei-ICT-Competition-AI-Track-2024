
import os
import argparse
import logging
import ast

import mindspore
import mindspore.nn as nn
from mindspore import Model, context
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.unet_medical import UNetMedical
from src.unet_nested import NestedUNet, UNet
from src.data_loader import create_dataset, create_multi_class_dataset
from src.loss import CrossEntropyWithLogits, MultiCrossEntropyWithLogits
from src.utils import StepLossTimeMonitor, UnetEval, TempLoss, apply_eval, filter_checkpoint_parameter_by_list, dice_coeff
from src.config import cfg_unet
from src.eval_callback import EvalCallBack

# device_id = int(os.getenv('DEVICE_ID'))
device_id=0

##1、请补充填写下面代码中缺失的部分
##-------------********------------
context.set_context(device_id = device_id, device_target='CPU')
##-------------********------------

mindspore.set_seed(1)


def train_net(args_opt,
              cross_valid_ind=1,
              epochs=400,
              batch_size=16,
              lr=0.01,
              cfg=None):
    rank = 0
    group_size = 1
    data_dir = args_opt.data_url
    run_distribute = args_opt.run_distribute

    if run_distribute:
        init()
        group_size = get_group_size()
        rank = get_rank()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          device_num=group_size,
                                          gradients_mean=False)
    need_slice = False
    if cfg['model'] == 'unet_medical':
        net = UNetMedical(n_channels=cfg['num_channels'], n_classes=cfg['num_classes'])
    elif cfg['model'] == 'unet_nested':
        net = NestedUNet(in_channel=cfg['num_channels'], n_class=cfg['num_classes'], use_deconv=cfg['use_deconv'],
                         use_bn=cfg['use_bn'], use_ds=cfg['use_ds'])
        need_slice = cfg['use_ds']
    elif cfg['model'] == 'unet_simple':
        net = UNet(in_channel=cfg['num_channels'], n_class=cfg['num_classes'])
    else:
        raise ValueError("Unsupported model: {}".format(cfg['model']))

    if cfg['resume']:
        param_dict = load_checkpoint(cfg['resume_ckpt'])
        if cfg['transfer_training']:
            filter_checkpoint_parameter_by_list(param_dict, cfg['filter_weight'])
        load_param_into_net(net, param_dict)

    if 'use_ds' in cfg and cfg['use_ds']:
        criterion = MultiCrossEntropyWithLogits()
    else:
        criterion = CrossEntropyWithLogits()
    if 'dataset' in cfg and cfg['dataset'] != "ISBI":
        repeat = cfg['repeat'] if 'repeat' in cfg else 1
        split = cfg['split'] if 'split' in cfg else 0.8

        # dataset_sink_mode = True
        # per_print_times = 30
        dataset_sink_mode = False
        per_print_times = 1

        train_dataset = create_multi_class_dataset(data_dir, cfg['img_size'], repeat, batch_size,
                                                   num_classes=cfg['num_classes'], is_train=True, augment=True,
                                                   split=split, rank=rank, group_size=group_size, shuffle=True)
        valid_dataset = create_multi_class_dataset(data_dir, cfg['img_size'], 1, 1,
                                                   num_classes=cfg['num_classes'], is_train=False,
                                                   eval_resize=cfg["eval_resize"], split=split,
                                                   python_multiprocessing=False, shuffle=False)
    else:
        repeat = cfg['repeat']
        dataset_sink_mode = False
        per_print_times = 1
        train_dataset, valid_dataset = create_dataset(data_dir, repeat, batch_size, True, cross_valid_ind,
                                                      run_distribute, cfg["crop"], cfg['img_size'])
    train_data_size = train_dataset.get_dataset_size()
    print("dataset length is:", train_data_size)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size,
                                   keep_checkpoint_max=cfg['keep_checkpoint_max'])
    ckpoint_cb = ModelCheckpoint(prefix='ckpt_{}_adam'.format(cfg['model']),
                                 directory='./ckpt_{}/'.format(device_id),
                                 config=ckpt_config)
    
    ##2、请自定义补充填写下面代码中缺失的部分，使用adam优化器
    ##-------------********------------
    #optimizer = xxx(params=net.xxxx(), learning_rate=, weight_decay=,
    #                    loss_scale=)
    optimizer = nn.Adam(params=net.trainable_params(),learning_rate=lr,weight_decay=cfg_unet['weight_decay'],
                        loss_scale=cfg_unet['loss_scale'])
    ##-------------********------------
    
    print("============== Starting Training ==============")
    
    #/3、请自定义补充填写下面代码中缺失的部分，定义训练部分
    ##-------------********------------
    
    for t in range(int(epochs / repeat)):
        print(f"Epoch {t}\n-------------------------------")
        train_loop(net,dataset=train_dataset,loss_fn=criterion,optimizer=optimizer)
        test_loop(net,valid_dataset,criterion)
    print("Done!")
    ##-------------********------------
    
    
    #/4、请自定义补充填写下面代码中缺失的部分，定义模型保存，保存成best.ckpt。
    ##-------------********------------
    mindspore.save_checkpoint(net,'best.ckpt')
    ##-------------********------------
    
    print("============== End Training ==============")


from mindspore import ops
def train_loop(model, dataset, loss_fn, optimizer):
    # Define forward function
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    #/5、请自定义补充填写下面代码，定义用于训练的train_loop函数，使用函数式自动微分，需先定义正向函数forward_fn，使用ops.value_and_grad获得微分函数grad_fn。然后，我们将微分函数和优化器的执行封装为train_step函数，接下来循环迭代数据集进行训练即可。
    ##-------------********------------
    #grad_fn = ops.xxx(xxx, None, xxx, has_aux=True)
    grad_fn = ops.value_and_grad(forward_fn,None,optimizer.parameters,has_aux=True)
    # Define function of one-step training
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss
    ##-------------********------------
    
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")



def test_loop(model, dataset, loss_fn):
    #/6、请自定义补充填写下面代码，定义用于用于测试的test_loop函数，test_loop函数同样需循环遍历数据集，调用模型计算loss并返回最终结果
    ##-------------********------------
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    test_loss = 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        test_loss += loss_fn(pred, label).asnumpy()
        
    test_loss /= num_batches
    
    print(f"Test: \n Avg loss: {test_loss:>8f} \n")
##-------------********------------
            
            
    
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_url', dest='data_url', type=str, default='data/',
                        help='data directory')
    parser.add_argument('-t', '--run_distribute', type=ast.literal_eval,
                        default=False, help='Run distribute, default: false.')
    parser.add_argument("--run_eval", type=ast.literal_eval, default=False,
                        help="Run evaluation when training, default is False.")
    parser.add_argument("--save_best_ckpt", type=ast.literal_eval, default=True,
                        help="Save best checkpoint when run_eval is True, default is True.")
    parser.add_argument("--eval_start_epoch", type=int, default=0,
                        help="Evaluation start epoch when run_eval is True, default is 0.")
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="Evaluation interval when run_eval is True, default is 1.")
    parser.add_argument("--eval_metrics", type=str, default="dice_coeff", choices=("dice_coeff", "iou"),
                        help="Evaluation metrics when run_eval is True, support [dice_coeff, iou], "
                             "default is dice_coeff.")

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    print("Training setting:", args)

    epoch_size = cfg_unet['epochs'] if not args.run_distribute else cfg_unet['distribute_epochs']
    train_net(args_opt=args,
              cross_valid_ind=cfg_unet['cross_valid_ind'],
              epochs=epoch_size,
              batch_size=cfg_unet['batchsize'],
              lr=cfg_unet['lr'],
              cfg=cfg_unet)
