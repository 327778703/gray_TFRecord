import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='FangSong')
import numpy as np
# tf.compat.v1.disable_eager_execution()

path = r'D:\MyFiles\ResearchSubject\Alldatasets3\Alldatasets3\valid'
cwd = path + '\\'
classes = os.listdir(path)

tfrecords_file = r'D:\MyFiles\ResearchSubject\Alldatasets3\gray_tfrecords2\valid.tfrecords'

dataset = tf.data.TFRecordDataset(tfrecords_file)
# for raw_record in dataset.take(2):
#     print(repr(raw_record))

features = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'img_raw': tf.io.FixedLenFeature([], tf.string)
}

def read_and_decode(example_string):
    features_dic = tf.io.parse_single_example(example_string, features)  # 解析example序列变成的字符串序列
    img = tf.io.decode_raw(features_dic['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 1])
    label = tf.cast(features_dic['label'], tf.int32)
    return img, label

from MeanStd import MeanStd
mean, std = MeanStd().Getmean_std()

def standardize(image_data):
    image_data = tf.cast(image_data, tf.float32)
    image_data = (image_data - mean)/std
    # 将RGB转成BGR，符合VGG16预训练模型的输入要求（预处理要求）
    # 在VGG16中预处理要求还有一条要进行中心化，但是如果采用VGG16默认的预处理方法，则中心化是以ImageNet数据集而言的，因此不能采用VGG16
    # 默认的预处理方法
    return image_data

# 展示TFRecord数据集图片
BATCH = 32
dataset = dataset.map(read_and_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(buffer_size=32000)
dataset = dataset.map(lambda x, y: (standardize(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=BATCH)
batch_dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def reverse_standardize(image_data):
    image_data = np.clip(image_data * std + mean, 0, 255)
    return image_data

# for parsed_record in parsed_dataset.take(4):
#     print(repr(parsed_record))  # 展示解析后的dataset-tensor

# i = 1
# for image, label in shuffled_dataset.take(1):
#     plt.figure(figsize=(2.56, 2.56))
#     plt.imshow(image)
#     plt.title(classes[label])
#     # plt.savefig('D:\\door_test\\' + classes[label] + '_' + str(i) + '.jpg')
#     plt.axis('off')
#     plt.show()
#     i += 1
# #
plt.figure(figsize=(10, 10))
for images, labels in batch_dataset.take(1):
    # 取了1个batch_size的数据
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(reverse_standardize(images[i]).reshape((256, 256)).astype('uint8'), cmap='gray')
        plt.title(classes[labels[i]])
        plt.axis('off')
plt.show()
