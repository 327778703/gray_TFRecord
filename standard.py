# -*- coding: utf-8 -*-
# 计算数据集的均值和标准差

import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib
matplotlib.rc("font", family='FangSong')
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np

# tensorflow版本
print("tf.version:", tf.__version__)

# 数据集获取
TRAIN_BATCH_SIZE = 64
# VALID_BATCH_SIZE = 32
# TEST_BATCH_SIZE = 32
IMG_SIZE = (256, 256)
train_path = r"D:\MyFiles\ResearchSubject\Alldatasets3\Alldatasets3\train"
# valid_path = r"D:\MyFiles\ResearchSubject\Alldatasets3\Alldatasets3\valid"
# test_path = r"D:\MyFiles\ResearchSubject\Alldatasets3\Alldatasets3\test"

train_dataset = image_dataset_from_directory(train_path, batch_size=TRAIN_BATCH_SIZE, image_size=IMG_SIZE, shuffle=True,
                                             color_mode='grayscale')
# valid_dataset = image_dataset_from_directory(valid_path, batch_size=VALID_BATCH_SIZE, image_size=IMG_SIZE, shuffle=True)
# test_dataset = image_dataset_from_directory(test_path, batch_size=TEST_BATCH_SIZE, image_size=IMG_SIZE, shuffle=True)
className = train_dataset.class_names  # 这里标签可以这样得到
for i in range(len(className)):
    c = re.split("_", className[i])
    className[i] = c[1]+"_"+c[2]
print("64个类：", className)

# 提前取好数据
AUTOTUNE = tf.data.AUTOTUNE
train_batch_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)



# 批次数据标准化
def standardize(image_data):
    mean, var = tf.nn.moments(image_data, axes=[0, 1, 2])  # 默认是ddof=0，使用总体标准差，也就是除以n
    # 使用总体标准差后得到的数据并不是以0为均值，使用样本标准差后得到的数据就是以0为均值，1为方差
    std = tf.math.sqrt(var)
    return mean, std

mean, std = tf.zeros([1, ]), tf.zeros([1, ])
for i in range(train_batch_dataset.cardinality()):
    imgs, labels = next(iter(train_batch_dataset))
    if len(labels) == TRAIN_BATCH_SIZE:
        mean1, std1 = standardize(imgs)
        mean = mean + (mean1 - mean) / (i+1)
        std = std + (std1 - std) / (i+1)
        print("""batch{}:
        mean1:{}, mean:{}
        std1:{}, std:{}
        -------------------""".format(i+1, mean1, mean, std1, std))
        print(mean1, mean)
        print(std1, std)
    else:
        mean1, std1 = standardize(imgs)
        mean = (mean*TRAIN_BATCH_SIZE*(train_batch_dataset.cardinality()-1) + len(labels)*mean1) / (TRAIN_BATCH_SIZE*(train_batch_dataset.cardinality()-1)+ len(labels)*mean1)
        std = (std*TRAIN_BATCH_SIZE*(train_batch_dataset.cardinality()-1) + len(labels)*std1) / (TRAIN_BATCH_SIZE*(train_batch_dataset.cardinality()-1)+ len(labels)*mean1)

print('train_datset, mean:{}, std:{}'.format(mean, std))
