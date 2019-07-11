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


# def safe_boolean_mask(input_tensor, mask_array):
#     single_row = input_tensor[0, :]
#     zero_dim = tf.zeros_like(single_row)
#     extended_x = tf.concat([input_tensor, zero_dim], axis=0)
#     extended_mask = tf.concat([mask_array, tf.ones(shape=(1, ))], axis=0)
#     masked_x = tf.boolean_mask(extended_x, extended_mask)
#
#
#

batch_size = 250
child_count = 3
channel_count = 32

dataTensor = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="dataTensor")
indices_tensor = tf.placeholder(name="indices_tensor", dtype=tf.int32)
batch_size_tensor = tf.placeholder(name="batch_size_tensor", dtype=tf.int32)

condition_indices_list = []
partition_list = []
mask_list = []
for child_index in range(child_count):
    mask_indices = tf.reshape(indices_tensor[:, child_index], [-1])
    condition_indices = tf.boolean_mask(tf.range(batch_size_tensor), mask_indices)
    partition = tf.boolean_mask(dataTensor, mask_indices)
    mask_list.append(mask_indices)
    condition_indices_list.append(condition_indices)
    partition_list.append(partition)

transformed_list = [build_conv_layer(input=part, filter_size=5, num_of_input_channels=1, num_of_output_channels=32)
                    for part in partition_list]
squared_list = [tf.square(part) for part in partition_list]
stitched_conv_transform = tf.dynamic_stitch(indices=condition_indices_list, data=transformed_list)
stitched_square_transform = tf.dynamic_stitch(indices=condition_indices_list, data=squared_list)
sum = tf.reduce_sum(stitched_square_transform)
grads = tf.gradients(sum, dataTensor)

sess = tf.Session()
samples = np.random.uniform(size=(batch_size, 28, 28, 1))
indices_arr = np.zeros(shape=(batch_size, child_count), dtype=np.int32)
indices_arr[:, 0] = 1
indices_arr[-2] = np.array([0, 1, 0])
indices_arr[-1] = np.array([0, 1, 0])

feed_dict = {dataTensor: samples,
             batch_size_tensor: batch_size,
             # indices_tensor: np.argmax(np.random.uniform(size=(GlobalConstants.EVAL_BATCH_SIZE, child_count)), axis=1)}
             indices_tensor: indices_arr}
outputs = []
outputs.extend(mask_list)
outputs.extend(transformed_list)
outputs.extend(squared_list)
outputs.append(stitched_conv_transform)
outputs.append(stitched_square_transform)
outputs.append(sum)
outputs.append(grads)

init = tf.global_variables_initializer()
sess.run(init)
for i in range(10000):
    results = sess.run(outputs, feed_dict=feed_dict)
    assert np.allclose(results[-1][0], 2.0*samples)
    print("{0} runned.".format(i))

