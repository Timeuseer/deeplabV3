#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About :
'''

import os
import yaml
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    config = yaml.load(open(r'./config/miou.yaml', 'r', encoding='utf_8_sig'))

    image_ids = open(os.path.join(config['VOCdevkit_path'], "VOC2007/ImageSets/Segmentation/val.txt"),
                     'r').read().splitlines()
    gt_dir = os.path.join(config['VOCdevkit_path'], "VOC2007/SegmentationClass/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    if config['miou_mode'] == 0 or config['miou_mode'] == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        deeplab = DeeplabV3()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(config['VOCdevkit_path'], "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = deeplab.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if config['miou_mode'] == 0 or config['miou_mode'] == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, config['num_classes'],
                                                        config['name_classes'])  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, config['name_classes'])
