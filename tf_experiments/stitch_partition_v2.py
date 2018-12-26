import tensorflow as tf
import numpy as np

from data_handling.fashion_mnist import FashionMnistDataSet
from simple_tf.global_params import GlobalConstants

dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
child_count = 3

dataTensor = tf.placeholder(GlobalConstants.DATA_TYPE,
                            shape=(None, dataset.get_image_size(),
                                   dataset.get_image_size(),
                                   dataset.get_num_of_channels()),
                            name="dataTensor")
batch_size_tensor = tf.placeholder(name="batch_size_tensor", dtype=tf.int32)

indices_tensor = tf.ones(shape=(batch_size_tensor, ))
condition_indices = tf.dynamic_partition(data=tf.range(batch_size_tensor), partitions=indices_tensor,
                                         num_partitions=child_count)
partition_list = tf.dynamic_partition(data=dataTensor, partitions=indices_tensor, num_partitions=child_count)

