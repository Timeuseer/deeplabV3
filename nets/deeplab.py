#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 
'''
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from nets.mobilenet import mobilenetV2


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    # 计算padding的数量，hw是否需要收缩
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = layers.Activation('relu')(x)

    '''
    分离卷积，首先进行3x3分离卷积，再进行1x1深度卷积
    '''
    # 3x3采用膨胀卷积
    x = layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                               padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = layers.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = layers.Activation('relu')(x)

    # 1x1卷积，进行压缩
    x = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise')(x)
    x = layers.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)

    if depth_activation:
        x = layers.Activation('relu')(x)

    return x


def Deeplabv3(input_shape, num_classes, alpha=1,backbone='mobilenet', downsample_factor=16):
    img_input = layers.Input(shape=input_shape)
    x, atrous_rates, skip1 = mobilenetV2(img_input, alpha, downsample_factor=downsample_factor)
    size_before = K.int_shape(x)

    '''
    总共五个分支
    ASPP实现特征提取
    利用不同膨胀率的膨胀卷积进行特征提取
    '''
    # 分支0
    b0 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = layers.Activation('relu', name='aspp0_relu')(b0)

    # 分支1 rate=6
    b1 = SepConv_BN(x, 256, 'aspp1', rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)

    # 分支2 rate=12
    b2 = SepConv_BN(x, 256, 'aspp2', rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)

    # 分支3 rate=18
    b3 = SepConv_BN(x, 256, 'aspp3', rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    # 分支4 全部求平均后，再利用expend_dims扩充维度，然后利用1x1调整通道
    b4 = layers.GlobalAveragePooling2D()(x)
    b4 = layers.Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = layers.Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = layers.BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = layers.Activation('relu')(b4)
    b4 = layers.Lambda(lambda x: tf.compat.v1.image.resize_images(x, size_before[1:3], align_corners=True))(b4)

    '''
    将五个分支堆叠
    再使用1x1卷积整合特征
    '''
    x = layers.Concatenate()([b4, b0, b1, b2, b3])
    # 利用卷积压缩
    x = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = layers.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)

    skip_size = K.int_shape(skip1)
    '''
    加强特征边上采样
    '''
    x = layers.Lambda(lambda xx: tf.compat.v1.image.resize_images(xx, skip_size[1:3], align_corners=True))(x)
    '''
    浅层特征边
    '''
    dec_skip1 = layers.Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = layers.BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = layers.Activation('relu')(dec_skip1)

    '''
    与浅层特征堆叠后利用卷积进行特征提取
    '''
    x = layers.Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 256, 'decoder_conv0', depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1', depth_activation=True, epsilon=1e-5)

    '''
    获得每个像素点的分类
    '''
    size_before3 = K.int_shape(img_input)
    x = layers.Conv2D(num_classes, (1, 1), padding='same')(x)
    x = layers.Lambda(lambda xx: tf.compat.v1.image.resize_images(xx, size_before3[1:3], align_corners=True))(x)
    x = layers.Softmax()(x)

    model = models.Model(img_input, x, name='deeplabv3plus')
    model.summary()
    return model
