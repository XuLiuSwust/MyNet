## 关于数据读取的api

> https://zhuanlan.zhihu.com/p/30751039
> 
> https://www.tensorflow.org/api_docs/python/tf/data

```python
filenames = get_filenames(is_training, data_dir)
# tf.data.TextLineDataset()：这个函数的输入是一个文件的列表，输出是一个dataset。
#   dataset中的每一个元素就对应了文件中的一行。可以使用这个函数来读入CSV文件。
# tf.data.FixedLengthRecordDataset()：这个函数的输入是一个文件的列表和一个
#   record_bytes，之后dataset的每一个元素就是文件中固定字节数record_bytes的内容。通
#   常用来读取以二进制形式保存的文件，如CIFAR10数据集就是这种形式。
# tf.data.TFRecordDataset()：顾名思义，这个函数是用来读TFRecord文件的，dataset中的
#   每一个元素就是一个TFExample。
dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
# 从数据里去拿出batchsize大小的数据
dataset = dataset.prefetch(buffer_size=batch_size)
```