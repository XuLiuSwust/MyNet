from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import config.cfg as cfg

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_NUM_IMAGES = {
  'train'     : 50000,
  'validation': 10000,
}


def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), (
    'Run cifar10_download_and_extract.py first to download and extract the '
    'CIFAR-10 data.')

  if is_training:
    return [
      os.path.join(data_dir, 'data_batch_%d.bin' % i)
      for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record, is_training):
  record_vector = tf.decode_raw(raw_record, tf.uint8)
  label = tf.cast(record_vector[0], tf.int32)
  depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                           [_NUM_CHANNELS, _HEIGHT, _WIDTH])

  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
  image = preprocess_image(image, is_training)
  return image, label


def preprocess_image(image, is_training):
  """
  图像预处理
  """
  if is_training:
    # 图像比目标大小大就剪裁, 小就padding到目标大小
    # 剪裁的时候, 从图像的中心进行的剪裁
    image = tf.image.resize_image_with_crop_or_pad(
      image, _HEIGHT + 8, _WIDTH + 8)
    # 随机选择位置进行剪裁
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])
    # 1/2的概率随机左右翻转
    image = tf.image.random_flip_left_right(image)
  # 标准化, 零均值, 单位方差, 输出大小和输入一样
  image = tf.image.per_image_standardization(image)
  return image


def input_fn(is_training, common_params, dataset_params):
  """
  获取文件, 读取数据,
  """
  data_dir = dataset_params['data_path']
  batch_size = common_params['batch_size']
  num_epochs = common_params['num_epochs']
  filenames = get_filenames(is_training, data_dir)
  # tf.data.TextLineDataset()：这个函数的输入是一个文件的列表，输出是一个dataset。
  #   dataset中的每一个元素就对应了文件中的一行。可以使用这个函数来读入CSV文件。
  # tf.data.FixedLengthRecordDataset()：这个函数的输入是一个文件的列表和一个
  #   record_bytes，之后dataset的每一个元素就是文件中固定字节数record_bytes的内容。通
  #   常用来读取以二进制形式保存的文件，如CIFAR10数据集就是这种形式。
  # tf.data.TFRecordDataset()：顾名思义，这个函数是用来读TFRecord文件的，dataset中的
  #   每一个元素就是一个TFExample。
  # 这里每条记录中的字节数是图像大小加上一比特
  dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
  # 从数据里去拿出batchsize大小的数据
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # 随机混淆数据后抽取buffer_size大小的数据
    dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

  # 将数据集重复周期次, 这么多周期都用使用相同的数据
  dataset = dataset.repeat(num_epochs)
  # 把转换函数应用到数据集上
  # map映射函数, 并使用batch操作进行批提取
  dataset = dataset.apply(tf.contrib.data.map_and_batch(
    lambda value: parse_record(value, is_training),
    batch_size=batch_size, num_parallel_batches=1, drop_remainder=False))

  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  return dataset


def cifar_dataset(common_params, dataset_params):
  """
  获取输入数据, 测试数据, 整理成数据集
  """
  # train是多了一个混淆抽取
  train_dataset = input_fn(True, common_params, dataset_params)
  test_dataset = input_fn(False, common_params, dataset_params)
  dataset = {
    'train': train_dataset,
    'test' : test_dataset
  }
  return dataset


def input_fn_test(dataset_params):
  data_dir = dataset_params['data_path']
  batch_size = _NUM_IMAGES['validation']
  filenames = get_filenames(False, data_dir)
  dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
  dataset = dataset.prefetch(buffer_size=batch_size)
  dataset = dataset.apply(tf.contrib.data.map_and_batch(
    lambda value: parse_record(value, False),
    batch_size=batch_size,
    num_parallel_batches=1,
    drop_remainder=False))

  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  return dataset


def cifar_dataset_test(dataset_params):
  dataset = input_fn_test(dataset_params)
  return dataset


if __name__ == '__main__':
  dataset = cifar_dataset_test(cfg.dataset_params)
  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()
  sess = tf.Session()
  images, labels = sess.run(next_element)
  print(images.shape)
  print(dataset)
