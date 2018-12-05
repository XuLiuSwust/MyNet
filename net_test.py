import os

import tensorflow as tf

import config.cfg as cfg


def parse_record(raw_record, is_training):
  record_vector = tf.decode_raw(raw_record, tf.uint8)
  label = tf.cast(record_vector[0], tf.int32)
  depth_major = tf.reshape(record_vector[1:cfg.RECORD_BYTES],
                           [cfg.NUM_CHANNELS, cfg.HEIGHT, cfg.WIDTH])

  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
  image = preprocess_image(image, is_training)
  return image, label


def preprocess_image(image, is_training):
  """
  图像预处理
  """
  if is_training:
    # 调整图像到固定大小
    # 剪裁的时候, 从图像的中心进行的剪裁
    image = tf.image.resize_image_with_crop_or_pad(
      image, cfg.HEIGHT + 8, cfg.WIDTH + 8)

    # 随机选择位置进行剪裁
    image = tf.random_crop(image, [cfg.HEIGHT, cfg.WIDTH, cfg.NUM_CHANNELS])

    # 1/2的概率随机左右翻转
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_flip_up_down(image)

    # 图像色彩调整 ##############################################################
    # 随机设置图片的亮度
    image = tf.image.random_brightness(image, max_delta=30)
    # 随机设置图片的对比度
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    # 随机设置图片的色度
    image = tf.image.random_hue(image, max_delta=0.3)
    # 随机设置图片的饱和度
    image = tf.image.random_saturation(image, lower=0.2, upper=1.8)

  # 标准化, 零均值, 单位方差, 输出大小和输入一样
  image = tf.image.per_image_standardization(image)
  return image

def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), (
    'Run cifar10_download_and_extract.py first to download and extract the '
    'CIFAR-10 data.')

  if is_training:
    return [
      os.path.join(data_dir, 'data_batch_%d.bin' % i)
      for i in range(1, cfg.NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def input_fn_test(dataset_params):
  data_dir = dataset_params['data_path']
  batch_size = cfg.NUM_IMAGES['validation']
  filenames = get_filenames(False, data_dir)
  dataset = tf.data.FixedLengthRecordDataset(filenames, cfg.RECORD_BYTES)
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


def main_test(_):
  model_folder = os.path.join(cfg.dataset_params['model_path'],
                              cfg.common_params['net_name'], 'ckpt')
  checkpoint = tf.train.get_checkpoint_state(model_folder)
  input_checkpoint = checkpoint.model_checkpoint_path
  saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                     clear_devices=True)

  dataset = cifar_dataset_test(cfg.dataset_params)
  test_iterator = dataset.make_one_shot_iterator()
  test_Loader = test_iterator.get_next()
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    saver.restore(sess, input_checkpoint)
    images, labels = sess.run(test_Loader)

    total_count = 0
    count = 1
    # 抽取数据集里的数据来进行评估
    for i in range(0, len(images), 50):
      image = images[i:i + 50, :, :, :]
      predict = sess.run(cfg.graph_node['output'],
                         feed_dict={cfg.graph_node['input']      : image,
                                    cfg.graph_node['is_training']: False,
                                    cfg.graph_node['keep_prob']  : 1.0})
      correct_pred = tf.equal(
        tf.argmax(predict, 1, output_type=tf.int32), labels[i:i + 50])
      acc = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
      total_count += sess.run(acc)
      accuracy = total_count / (count * 50)
      print("test samples:%d,accuracy:%d/%d = %.4f " % (
        count * 50, total_count, count * 50, accuracy))
      count += 1


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main_test)
