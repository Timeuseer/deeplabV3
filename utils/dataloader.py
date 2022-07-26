#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 
'''
import math
import os
from random import shuffle

import cv2
import numpy as np
from PIL import Image
from tensorflow import keras

from utils.utils import cvtColor, preprocess_input


class DeeplabDataset(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, batch_size, num_classes, train, dataset_path):
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        images = []
        targets = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % self.length
            name = self.annotation_lines[i].split()[0]
            '''
            从文件中读取图像
            '''
            jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
            png = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
            '''
            数据增强
            '''
            jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)
            jpg = preprocess_input(np.array(jpg, np.float32))
            png = np.array(png)
            png[png >= self.num_classes] = self.num_classes
            '''
            转化为one_hot
            '''
            seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
            seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

            images.append(jpg)
            targets.append(seg_labels)

        images = np.array(images)
        targets = np.array(targets)
        return images, targets

    def on_epoch_end(self):
        shuffle(self.annotation_lines)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        h, w = input_shape

        if not random:
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))

            return new_image, new_label

        rand_jit1 = self.rand(1 - jitter, 1 + jitter)
        rand_jit2 = self.rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2

        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)

        '''
        高斯平滑
        '''
        blur = self.rand() < .25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        '''
        旋转
        '''
        rotate = self.rand() < .25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        '''
        色域变换
        '''
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        '''
        将图像转到HSV上
        '''
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label
