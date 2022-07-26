#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 
'''
import colorsys
import copy
import time
import cv2
import numpy as np
import tensorflow as tf
import yaml
from PIL import Image
from nets.deeplab import Deeplabv3
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


class DeeplabV3(object):
    def __init__(self):
        config = yaml.load(open(r'./config/deeplab.yaml', 'r', encoding='utf_8_sig'))
        self.__dict__.update(config)
        '''
        设置框的颜色
        '''
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128),
                           (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        '''
        加载模型
        '''
        self.generate()

        show_config(**config)

    '''
    获得所有的分类
    '''

    def generate(self):
        '''
        加载模型和权重
        '''
        self.model = Deeplabv3([self.input_shape[0], self.input_shape[1], 3], self.num_classes,
                               backbone=self.backbone, downsample_factor=self.downsample_factor)
        self.model.load_weights(self.model_path)
        print(f'{self.model_path} model loaded.')

    @tf.function
    def get_pred(self, image_data):
        pr = self.model(image_data, training=False)

        return pr

    # 检测图片
    def detect_image(self, image, count=False, name_classes=None):
        '''
         保证为rgb图像
        '''
        image = cvtColor(image)
        '''
        对输入输入图像进行备份
        '''
        old_img = copy.deepcopy(image)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        image_data = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        '''
        传入模型进行预测
        '''
        pr = self.get_pred(image_data)[0].numpy()

        '''
        去掉灰条部分
        '''
        pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
             int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        '''
        取出每一个像素点的种类
        '''
        pr = pr.argmax(axis=-1)

        '''
        计数
        '''
        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = original_h * original_w
            print('-' * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num

            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [original_h, original_w, -1])
            image = Image.fromarray(np.uint8(seg_img))
            image = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image = Image.fromarray(np.uint8(seg_img))

        return image

    def get_FPS(self, image, test_interval):
        '''
                 保证为rgb图像
                '''
        image = cvtColor(image)
        '''
        对输入输入图像进行备份
        '''
        old_img = copy.deepcopy(image)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        image_data = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        '''
        传入模型进行预测
        '''
        pr = self.get_pred(image_data)[0].numpy()

        '''
        去掉灰条部分
        '''
        pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
             int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        '''
        取出每一个像素点的种类
        '''
        pr = pr.argmax(axis=-1)

        t1 = time.time()
        for _ in range(test_interval):
            pr = self.get_pred(image_data)[0].numpy()
            pr = pr.argmax(axis=-1).reshape([self.input_shape[0], self.input_shape[1]])
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval

        return tact_time

    def get_miou_png(self, image):
        image = cvtColor(image)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        '''
        传入图片进行预测
        '''
        pr = self.get_pred(image_data)[0].numpy()

        pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
             int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image
