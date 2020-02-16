import tensorflow as tf
import numpy as np


# class FastRcnn:
#     def __init__(self):

imageInputs = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='input')

net = imageInputs
in_filters = imageInputs.get_shape().as_list()[-1]
out_filters = 32
pooled_height = 7
pooled_width = 7

W = tf.get_variable("W", [3, 3, in_filters, out_filters], trainable=True)
b = tf.get_variable("b", [out_filters], trainable=True)
net = tf.nn.conv2d(net, W, strides = [1, 2, 2, 1], padding='SAME')
net = tf.nn.bias_add(net, b)
net = tf.nn.relu(net)

X = np.random.uniform(low=0.0, high=1.0, size=(3, 2500, 640, 3))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

results = sess.run([net], feed_dict={imageInputs: X})

print("X")