#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About :
'''
from tensorflow.keras import activations, layers


def _make_divisible(v, divisor, min_value=None):
    '''
    用来限制通道数能够被8整除
    '''
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


def relu6(x):
    return activations.relu(x, max_value=6)


def _inverted_res_block(inputs, expansion, stride, alpha, in_filters, filters, block_id, skip_connection, rate=1):
    point_wise_filters = _make_divisible(int(filters * alpha), 8)
    prefix = f'expanded_conv_{block_id}_'

    x = inputs
    '''
    先利用1x1卷积进行通道数上升
    '''
    if block_id:
        x = layers.Conv2D(expansion * in_filters, kernel_size=1, padding='same',
                          use_bias=False, activation=None, name=prefix + 'expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                                      name=prefix + 'expand_BN')(x)
        x = layers.Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    '''
    利用可深度分离卷积进行特征提取
    '''
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                               use_bias=False, padding='same', dilation_rate=(rate, rate),
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    x = layers.Activation(relu6, name=prefix + 'depthwise_relu')(x)

    '''
    利用1x1的卷积进行通道数下降
    '''
    x = layers.Conv2D(point_wise_filters,
                      kernel_size=1, padding='same', use_bias=False, activation=None,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                                  name=prefix + 'project_BN')(x)

    '''
    添加残差边
    '''
    if skip_connection:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def mobilenetV2(inputs, alpha=1, downsample_factor=8):
    if downsample_factor == 8:
        block4_dilation = 2
        block5_dilation = 4
        block4_stride = 1
        atrous_rates = (12, 24, 36)
    elif downsample_factor == 16:
        block4_dilation = 1
        block5_dilation = 2
        block4_stride = 2
        atrous_rates = (6, 12, 18)
    else:
        raise ValueError('Unsupported factor - `{}`, Use 8 or 16.'.format(downsample_factor))

    first_block_filters = _make_divisible(32 * alpha, 8)
    # [512,512,3]->[256,256,32]
    x = layers.Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(2, 2), padding='same',
                      use_bias=False, name='Conv')(inputs)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = layers.Activation(relu6, name='Conv_Relu6')(x)

    x = _inverted_res_block(x, in_filters=32, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)

    # [256,256,16]->[128,128,24]
    x = _inverted_res_block(x, in_filters=16, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, in_filters=24, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)
    skip1 = x

    # [128,128,24]->[64,64,32]
    x = _inverted_res_block(x, in_filters=24, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, in_filters=32, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, in_filters=32, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)

    # [64,64,32]->[32,32,64]
    x = _inverted_res_block(x, in_filters=32, filters=64, alpha=alpha, stride=block4_stride,
                            expansion=6, block_id=6, skip_connection=False)
    x = _inverted_res_block(x, in_filters=64, filters=64, alpha=alpha, stride=1, rate=block4_dilation,
                            expansion=6, block_id=7, skip_connection=True)
    x = _inverted_res_block(x, in_filters=64, filters=64, alpha=alpha, stride=1, rate=block4_dilation,
                            expansion=6, block_id=8, skip_connection=True)
    x = _inverted_res_block(x, in_filters=64, filters=64, alpha=alpha, stride=1, rate=block4_dilation,
                            expansion=6, block_id=9, skip_connection=True)

    # [32,32,64]->[32,32,96]
    x = _inverted_res_block(x, in_filters=64, filters=96, alpha=alpha, stride=1, rate=block4_dilation,
                            expansion=6, block_id=10, skip_connection=False)
    x = _inverted_res_block(x, in_filters=96, filters=96, alpha=alpha, stride=1, rate=block4_dilation,
                            expansion=6, block_id=11, skip_connection=True)
    x = _inverted_res_block(x, in_filters=96, filters=96, alpha=alpha, stride=1, rate=block4_dilation,
                            expansion=6, block_id=12, skip_connection=True)

    # [32,32,96]->[32,32,160]->[32,32,320]
    x = _inverted_res_block(x, in_filters=96, filters=160, alpha=alpha, stride=1, rate=block4_dilation,  # 1!
                            expansion=6, block_id=13, skip_connection=False)
    x = _inverted_res_block(x, in_filters=160, filters=160, alpha=alpha, stride=1, rate=block5_dilation,
                            expansion=6, block_id=14, skip_connection=True)
    x = _inverted_res_block(x, in_filters=160, filters=160, alpha=alpha, stride=1, rate=block5_dilation,
                            expansion=6, block_id=15, skip_connection=True)

    x = _inverted_res_block(x, in_filters=160, filters=320, alpha=alpha, stride=1, rate=block5_dilation,
                            expansion=6, block_id=16, skip_connection=False)

    return x, atrous_rates, skip1
