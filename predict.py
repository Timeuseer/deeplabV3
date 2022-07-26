#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 
'''
import time
import yaml
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from deeplab import DeeplabV3

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':
    deeplab = DeeplabV3()
    config = yaml.load(open(r'config/predict.yaml', 'r', encoding='utf_8_sig'))

    if config['mode'] == 'predict':
        while True:
            img = input('Input image filepath:')
            try:
                image = Image.open(img)
            except:
                print('Open Error!')
                continue
            else:
                r_img = deeplab.detect_image(image, count=config['count'], name_classes=config['name_classes'])
                r_img.show()
    elif config['mode'] == 'video':
        capture = cv2.VideoCapture(config['video_path'])
        if config['video_save_path'] != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(config['video_save_path'], fourcc, config['video_fps'], size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while True:
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 转换格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变为Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(deeplab.detect_image(frame))
            # 转化为opencv格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print(f'fps = {fps:.2f}')
            frame = cv2.putText(frame, f'fps = {fps:.2f}', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('video', frame)
            c = cv2.waitKey(1) & 0xff
            if config['video_save_path'] != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        print('Video Detection Done!')
        capture.release()
        if config['video_save_path'] != '':
            print(f'Save processed video to the path : {config["video_save_path"]}')
            out.release()

        cv2.destroyWindows()
    elif config['mode'] == "fps":
        img = Image.open(config['fps_image_path'])
        tact_time = deeplab.get_FPS(img, config['test_interval'])
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif config['mode'] == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(config['dir_origin_path'])
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(config['dir_origin_path'], img_name)
                image = Image.open(image_path)
                r_image = deeplab.detect_image(image)
                if not os.path.exists(config['dir_save_path']):
                    os.makedirs(config['dir_save_path'])
                r_image.save(os.path.join(config['dir_save_path'], img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
