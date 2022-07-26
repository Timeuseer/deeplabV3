#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About :
'''
import tensorflow.keras.backend as K
import yaml
import os
import numpy as np
import datetime
import tensorflow as tf
from functools import partial
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from nets.deeplab import Deeplabv3
from nets.deeplab_training import (CE, Focal_Loss, dice_loss_with_CE,
                                   dice_loss_with_Focal_loss, get_lr_scheduler)
from utils.callbacks import EvalCallback, LossHistory, ModelCheckpoint
from utils.dataloader import DeeplabDataset
from utils.utils import show_config
from utils.utils_fit import fit_one_epoch
from utils.utils_metrics import Iou_score, f_score

if __name__ == '__main__':
    config = yaml.load(open(r'./config/train.yaml', 'r', encoding='utf_8_sig'))
    # 是否给不同类型赋予不同的损失权重
    cls_weights = np.ones([config['num_classes']], np.float32)

    '''
    设置用到的显卡
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in config['train_gpu'])
    ngpus_per_node = len(config['train_gpu'])

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    '''
    判断设置的GPU数量与实际的GPU数量是否合理
    '''
    if ngpus_per_node > 1 and ngpus_per_node > len(gpus):
        raise ValueError("The number of GPUs specified for training is more than the GPUs on the machine")

    if ngpus_per_node > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    print(f'Number of devices:{ngpus_per_node}')

    if ngpus_per_node > 1:
        with strategy.scope():
            '''
            获取model
            '''
            model = Deeplabv3([config['input_shape[0'], config['input_shape'][1], 3], config['num_classes'],
                              downsample_factor=config['downsample_factor'])
            if config['model_path'] != '':
                '''
                加载预训练权重
                '''
                model.load_weights(config['model_path'], by_name=True, skip_mismatch=True)
                print(f'Loaded weights {config["model_path"]}.')
    else:
        '''
        获取model
        '''
        model = Deeplabv3([config['input_shape'][0], config['input_shape'][1], 3], config['num_classes'],
                          downsample_factor=config['downsample_factor'])
        if config['model_path'] != '':
            '''
            加载预训练权重
            '''
            model.load_weights(config['model_path'], by_name=True, skip_mismatch=True)
            print(f'Loaded weights {config["model_path"]}.')

    '''
    使用到的损失函数
    '''
    if config['focal_loss']:
        if config['dice_loss']:
            loss = dice_loss_with_Focal_loss(cls_weights)
        else:
            loss = Focal_Loss(cls_weights)
    else:
        if config['dice_loss']:
            loss = dice_loss_with_CE(cls_weights)
        else:
            loss = CE(cls_weights)

    '''
    加载数据集
    '''
    with open(os.path.join(config['VOCdevkit_path'], "VOC2007/ImageSets/Segmentation/train.txt"), 'r') as f:
        train_lines = f.readlines()
    with open(os.path.join(config['VOCdevkit_path'], "VOC2007/ImageSets/Segmentation/val.txt"), 'r') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        num_classes=config['num_classes'], backbone=config['backbone'], model_path=config['model_path'],
        input_shape=config['input_shape'], \
        Init_Epoch=config['Init_Epoch'], Freeze_Epoch=config['Freeze_Epoch'], UnFreeze_Epoch=config['UnFreeze_Epoch'],
        Freeze_batch_size=config['Freeze_batch_size'], Unfreeze_batch_size=config['UnFreeze_batch_size'],
        Freeze_Train=config['Freeze_Train'], \
        Init_lr=config['Init_lr'], Min_lr=config['Init_lr'] * 1e-2, optimizer_type=config['optimizer_type'],
        momentum=config['momentum'], lr_decay_type=config['lr_decay_type'], \
        save_period=config['save_period'], save_dir=config['save_dir'], num_workers=config['num_workers'],
        num_train=num_train, num_val=num_val
    )

    wanted_step = 1.5e4 if config['optimizer_type'] == 'sgd' else 0.5e4
    total_step = num_train // config['UnFreeze_batch_size'] * config['UnFreeze_Epoch']

    if total_step <= wanted_step:
        if num_train // config['UnFreeze_batch_size'] == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        wanted_epoch = wanted_step // (num_train // config['UnFreeze_batch_size']) + 1

    '''
    主干特征提取网络特征通用，冻结训练可以加快训练速度
    '''
    if True:
        if config['Freeze_Train']:
            if config['backbone'] == 'mobilenet':
                freeze_layers = 146
            else:
                freeze_layers = 358

            for i in range(freeze_layers):
                model.layers[i].trainable = False
            print(f'Freeze the first {freeze_layers} layers of total {model.layers} layers.')

        '''
        如果不冻结训练，则设置batch_size为UnFreeze_batch_size
        '''
        batch_size = config['Freeze_batch_size'] if config['Freeze_Train'] else config['UnFreeze_batch_size']

        '''
        通过batch_size，自适应调整学习率
        '''
        nbs = 16
        lr_limit_max = 5e-4 if config['optimizer_type'] == 'adam' else 1e-1
        lr_limit_min = 3e-4 if config['optimizer_type'] == 'adam' else 5e-4
        if config['backbone'] == 'xception':
            lr_limit_max = 1e-4 if config['optimizer_type'] == 'adam' else 1e-1
            lr_limit_min = 1e-4 if config['optimizer_type'] == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * config['Init_lr'], lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * (config['Init_lr'] * 1e-2), lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        '''
        获得学习率下降公式
        '''
        lr_scheduler_func = get_lr_scheduler(config['lr_decay_type'], Init_lr_fit, Min_lr_fit, config['UnFreeze_Epoch'])

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        train_dataloader = DeeplabDataset(train_lines, config['input_shape'], batch_size, config['num_classes'], True,
                                          config['VOCdevkit_path'])
        val_dataloader = DeeplabDataset(val_lines, config['input_shape'], batch_size, config['num_classes'], False,
                                        config['VOCdevkit_path'])

        optimizer = {
            'adam': optimizers.Adam(lr=config['Init_lr'], beta_1=config['momentum']),
            'sgd': optimizers.SGD(lr=config['Init_lr'], momentum=config['momentum'], nesterov=True)
        }[config['optimizer_type']]

        if config['eager']:
            start_epoch = config['Init_Epoch']
            end_epoch = config['UnFreeze_Epoch']
            config['UnFreeze_flag'] = False

            gen = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

            gen = gen.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)
            gen_val = gen_val.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)

            if ngpus_per_node > 1:
                gen = strategy.experimental_distribute_dataset(gen)
                gen_val = strategy.experimental_distribute_dataset(gen_val)

            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
            log_dir = os.path.join(config['save_dir'], "loss_" + str(time_str))
            loss_history = LossHistory(log_dir)
            eval_callback = EvalCallback(model, config['input_shape'], config['num_classes'], val_lines,
                                         config['VOCdevkit_path'], log_dir, eval_flag=config['eval_flag'],
                                         period=config['eval_period'])
            '''
            开始模型训练
            '''
            for epoch in range(start_epoch, end_epoch):
                '''
                若模型有冻结学习，则解冻，并设置参数
                '''
                if epoch >= config['Freeze_Epoch'] and not config['UnFreeze_flag'] and config['Freeze_Train']:
                    batch_size = config['UnFreeze_batch_size']

                    '''
                    通过batch_size，自适应调整学习率
                    '''
                    nbs = 16
                    lr_limit_max = 5e-4 if config['optimizer_type'] == 'adam' else 1e-1
                    lr_limit_min = 3e-4 if config['optimizer_type'] == 'adam' else 5e-4
                    if config['backbone'] == 'xception':
                        lr_limit_max = 1e-4 if config['optimizer_type'] == 'adam' else 1e-1
                        lr_limit_min = 1e-4 if config['optimizer_type'] == 'adam' else 5e-4
                    Init_lr_fit = min(max(batch_size / nbs * config['Init_lr'], lr_limit_min), lr_limit_max)
                    Min_lr_fit = min(max(batch_size / nbs * (config['Init_lr'] * 1e-2), lr_limit_min * 1e-2),
                                     lr_limit_max * 1e-2)

                    '''
                    获得学习率下降公式
                    '''
                    lr_scheduler_func = get_lr_scheduler(config['lr_decay_type'], Init_lr_fit, Min_lr_fit,
                                                         config['UnFreeze_Epoch'])

                    for i in range(len(model.layers)):
                        model.layers[i].trainable = True

                    epoch_step = num_train // batch_size
                    epoch_step_val = num_val // batch_size

                    if epoch_step == 0 or epoch_step_val == 0:
                        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

                    train_dataloader.batch_size = batch_size
                    val_dataloader.batch_size = batch_size

                    gen = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
                    gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32))

                    gen = gen.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)
                    gen_val = gen_val.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)

                    if ngpus_per_node > 1:
                        gen = strategy.experimental_distribute_dataset(gen)
                        gen_val = strategy.experimental_distribute_dataset(gen_val)

                    config['UnFreeze_flag'] = True
                lr = lr_scheduler_func(epoch)
                K.set_value(optimizer.lr, lr)

                fit_one_epoch(model, loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
                              gen, gen_val, end_epoch, f_score(), config['save_period'], config['save_dir'], strategy)

                train_dataloader.on_epoch_end()
                val_dataloader.on_epoch_end()
        else:
            start_epoch = config['Init_Epoch']
            end_epoch = config['Freeze_Epoch'] if config['Freeze_Train'] else config['UnFreeze_Epoch']

            if ngpus_per_node > 1:
                with strategy.scope():
                    model.compile(loss=loss, optimizer=optimizer, metrics=[f_score()])
            else:
                model.compile(loss=loss, optimizer=optimizer, metrics=[f_score()])

            '''
            训练参数设置
            logging         tensorboard的保存地址
            checkpoint      保存权重细节
            lr_scheduler    设置学习率下降方式
            early_stopping  val_loss多次不下降，自动停止训练
            '''
            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
            log_dir = os.path.join(config['save_dir'], "loss_" + str(time_str))
            logging = callbacks.TensorBoard(log_dir)
            loss_history = LossHistory(log_dir)

            checkpoint = ModelCheckpoint(
                os.path.join(config['save_dir'], 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h'),
                monitor='val_loss', save_weights_only=True, save_best_only=False, period=config['save_period'])
            checkpoint_last = ModelCheckpoint(os.path.join(config['save_dir'], 'last_epoch_weights.h5'),
                                              monitor='val_loss', save_weights_only=True, save_best_only=False,
                                              period=1)
            checkpoint_best = ModelCheckpoint(os.path.join(config['save_dir'], 'best_epoch_weights.h5'),
                                              monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
            early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
            lr_scheduler = callbacks.LearningRateScheduler(lr_scheduler_func, verbose=1)
            eval_callback = EvalCallback(model, config['input_shape'], config['num_classes'], val_lines,
                                         config['VOCdevkit_path'], log_dir, eval_flag=config['eval_flag'],
                                         period=config['eval_period'])
            callback = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler,
                        eval_callback]

            if start_epoch < end_epoch:
                print(f'Train on {num_train} samples, val on {num_val} samples, with batch size {batch_size}.')
                model.fit(
                    x=train_dataloader,
                    steps_per_epoch=epoch_step,
                    validation_data=val_dataloader,
                    validation_steps=epoch_step_val,
                    epochs=end_epoch,
                    initial_epoch=start_epoch,
                    use_multiprocessing=True if config['num_workers'] > 1 else False,
                    workers=config['num_workers'],
                    callbacks=callback
                )

            '''
            如果模型有冻结学习部分，则解冻，并设置参数
            '''
            if config['Freeze_Train']:
                batch_size = config['UnFreeze_batch_size']
                start_epoch = config['Freeze_Epoch'] if start_epoch < config['Freeze_Epoch'] else start_epoch
                end_epoch = config['UnFreeze_Epoch']

                '''
                通过batch_size，自适应调整学习率
                '''
                nbs = 16
                lr_limit_max = 5e-4 if config['optimizer_type'] == 'adam' else 1e-1
                lr_limit_min = 3e-4 if config['optimizer_type'] == 'adam' else 5e-4
                if config['backbone'] == 'xception':
                    lr_limit_max = 1e-4 if config['optimizer_type'] == 'adam' else 1e-1
                    lr_limit_min = 1e-4 if config['optimizer_type'] == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * config['Init_lr'], lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * (config['Init_lr'] * 1e-2), lr_limit_min * 1e-2),
                                 lr_limit_max * 1e-2)

                '''
                获得学习率下降的公式
                '''
                lr_scheduler_func = get_lr_scheduler(config['lr_decay_type'], Init_lr_fit, Min_lr_fit,
                                                     config['UnFreeze_Epoch'])
                lr_scheduler = callbacks.LearningRateScheduler(lr_scheduler_func, verbose=1)
                callback = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler,
                            eval_callback]

                for i in range(len(model.layers)):
                    model.layers[i].trainable = True
                if ngpus_per_node > 1:
                    with strategy.scope():
                        model.compile(loss=loss,
                                      optimizer=optimizer,
                                      metrics=[f_score()])
                else:
                    model.compile(loss=loss,
                                  optimizer=optimizer,
                                  metrics=[f_score()])

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                train_dataloader.batch_size = config['UnFreeze_batch_size']
                val_dataloader.batch_size = config['UnFreeze_batch_size']

                print(f'Train on {num_train} samples, val on {num_val} samples, with batch size {batch_size}.')
                model.fit(
                    x=train_dataloader,
                    steps_per_epoch=epoch_step,
                    validation_data=val_dataloader,
                    validation_steps=epoch_step_val,
                    epochs=end_epoch,
                    initial_epoch=start_epoch,
                    use_multiprocessing=True if config['num_workers'] > 1 else False,
                    workers=config['num_workers'],
                    callbacks=callback
                )
