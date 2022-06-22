import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from PIL import Image
import numpy as np
# tf.compat.v1.disable_eager_execution()
#  # 把我们自己的数据集转成TFRecord


path = r'D:\MyFiles\ResearchSubject\Alldatasets3\Alldatasets3\valid'
cwd = path + '\\'
classes = os.listdir(path)

path = r'D:\MyFiles\ResearchSubject\Alldatasets3\gray_tfrecords2'
os.makedirs(path, exist_ok=True)
writer = tf.io.TFRecordWriter(path + r'\valid.tfrecords')
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

for index, name in enumerate(classes):
    print("now:", classes[index])
    class_path = cwd + name + '\\'
    NUM = 1
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        img = Image.open(img_path)
        img = img.resize((256, 256))
        img = img.convert('L')
        # imgArray = np.stack((img,) * 3, axis=-1)
        # img = Image.fromarray(imgArray)
        # img = tf.image.rgb_to_grayscale(img)
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": _int64_feature(index),
            "img_raw": _bytes_feature(img_raw)
        }))
        writer.write(example.SerializeToString())  # 将example序列变成字符串序列
        print('Creating record in ', NUM)
        NUM += 1
writer.close()



