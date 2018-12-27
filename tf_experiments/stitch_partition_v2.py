import tensorflow as tf
import numpy as np


def build_conv_layer(input, filter_size, num_of_input_channels, num_of_output_channels, name_suffix=""):
    # OK
    conv_weights = tf.Variable(
        tf.truncated_normal([filter_size, filter_size, num_of_input_channels, num_of_output_channels],
                            stddev=0.1, dtype=tf.float32))
    # OK
    conv_biases = tf.Variable(
        tf.constant(0.1, shape=[num_of_output_channels], dtype=tf.float32))
    conv = tf.nn.conv2d(input, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return pool

batch_size = 250
child_count = 3
channel_count = 32

dataTensor = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="dataTensor")
indices_tensor = tf.placeholder(name="indices_tensor", dtype=tf.int32)
batch_size_tensor = tf.placeholder(name="batch_size_tensor", dtype=tf.int32)

condition_indices = tf.dynamic_partition(data=tf.range(batch_size_tensor), partitions=indices_tensor,
                                         num_partitions=child_count)
partition_list = tf.dynamic_partition(data=dataTensor, partitions=indices_tensor, num_partitions=child_count)
transformed_list = [build_conv_layer(input=part, filter_size=5, num_of_input_channels=1,
                                     num_of_output_channels=channel_count)
                    for part in partition_list]
stitched = tf.dynamic_stitch(indices=condition_indices, data=transformed_list)

sess = tf.Session()
samples = np.random.uniform(size=(batch_size, 28, 28, 1))
indices_arr = np.zeros(shape=(batch_size, ), dtype=np.int32)
indices_arr[-1] = 2
indices_arr[-2] = 1
feed_dict = {dataTensor: samples,
             batch_size_tensor: batch_size,
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

