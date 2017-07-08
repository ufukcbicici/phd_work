import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

from data_handling.mnist_data_set import MnistDataSet




def baseline_network(node):




# dataset = MnistDataSet(validation_sample_count=5000)
# dataset.load_dataset()
# total_samples_seen = 0
#
# W1_shape = [5, 5, 1, 32]
# b1_shape = [32]
#
# x = tf.placeholder(tf.float32)
# initial_W1 = tf.truncated_normal(shape=W1_shape, stddev=0.1)
# W1 = tf.Variable(tf.truncated_normal(shape=W1_shape, stddev=0.1))
# initial_b1 = tf.constant(0.1, shape=b1_shape)
# b1 = tf.Variable(initial_b1)
# conv1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')
# conv1_sum = conv1 + b1
# # y = tf.placeholder(tf.float32)
# # z = conv1 + y
# # dif = z - conv1
#
#
# sess = tf.Session()
#
# # Run init ops
# init = tf.global_variables_initializer()
# sess.run(init)
#
# while True:
#     samples, labels, indices = dataset.get_next_batch(batch_size=1000)
#     samples = samples.reshape((1000, MnistDataSet.MNIST_SIZE, MnistDataSet.MNIST_SIZE, 1))
#     conv1_sum_res, W1_res, initial_W1_res = sess.run([conv1_sum, W1, initial_W1], feed_dict={x: samples})
#     print(W1_res[0, 0, 0, 0])
#     print(initial_W1_res[0, 0, 0, 0])
#     if dataset.isNewEpoch:
#         break
#
# print("X")
