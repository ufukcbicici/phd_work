import tensorflow as tf
import numpy as np

from data_handling.fashion_mnist import FashionMnistDataSet
from simple_tf.global_params import GlobalConstants


def build_conv_layer(input, filter_size, num_of_input_channels, num_of_output_channels, name_suffix=""):
    # OK
    conv_weights = tf.Variable(
        tf.truncated_normal([filter_size, filter_size, num_of_input_channels, num_of_output_channels],
                            stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE))
    # OK
    conv_biases = tf.Variable(
        tf.constant(0.1, shape=[num_of_output_channels], dtype=GlobalConstants.DATA_TYPE))
    conv = tf.nn.conv2d(input, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return pool


dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
child_count = 3

dataTensor = tf.placeholder(GlobalConstants.DATA_TYPE,
                            shape=(None, dataset.get_image_size(),
                                   dataset.get_image_size(),
                                   dataset.get_num_of_channels()),
                            name="dataTensor")
indices_tensor = tf.placeholder(name="indices_tensor", dtype=tf.int32)
batch_size_tensor = tf.placeholder(name="batch_size_tensor", dtype=tf.int32)

condition_indices = tf.dynamic_partition(data=tf.range(batch_size_tensor), partitions=indices_tensor,
                                         num_partitions=child_count)
partition_list = tf.dynamic_partition(data=dataTensor, partitions=indices_tensor, num_partitions=child_count)
transformed_list = [build_conv_layer(input=part, filter_size=5, num_of_input_channels=1, num_of_output_channels=32)
                    for part in partition_list]
stitched = tf.dynamic_stitch(indices=condition_indices, data=transformed_list)

sess = tf.Session()
minibatch = dataset.get_next_batch(batch_size=GlobalConstants.EVAL_BATCH_SIZE)
indices_arr = np.zeros(shape=(GlobalConstants.EVAL_BATCH_SIZE, ), dtype=np.int32)
indices_arr[-1] = 1
indices_arr[-2] = 2
feed_dict = {dataTensor: minibatch.samples,
             batch_size_tensor: GlobalConstants.EVAL_BATCH_SIZE,
             # indices_tensor: np.argmax(np.random.uniform(size=(GlobalConstants.EVAL_BATCH_SIZE, child_count)), axis=1)}
             indices_tensor: indices_arr}
outputs = []
outputs.extend(transformed_list)
outputs.append(stitched)

init = tf.global_variables_initializer()
sess.run(init)
for i in range(10000):
    results = sess.run(outputs, feed_dict=feed_dict)
    print("{0} runned.".format(i))

