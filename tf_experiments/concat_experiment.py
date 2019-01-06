import tensorflow as tf
import numpy as np

from data_handling.mnist_data_set import MnistDataSet

k = 3
D = MnistDataSet.MNIST_SIZE * MnistDataSet.MNIST_SIZE
threshold = 0.3
feature_count = 32
epsilon = 0.000001
batch_size = 100

dataset = MnistDataSet(validation_sample_count=5000)
dataset.load_dataset()

samples, labels, indices_list = dataset.get_next_batch()
index_list = np.arange(0, batch_size)
initializer = tf.contrib.layers.xavier_initializer()
x = tf.placeholder(tf.float32, name="x")
indices = tf.placeholder(tf.int64, name="indices")
# Convolution
x_image = tf.reshape(x, [-1, MnistDataSet.MNIST_SIZE, MnistDataSet.MNIST_SIZE, 1])
C = tf.get_variable(name="C", shape=[5, 5, 1, feature_count], initializer=initializer,
                    dtype=tf.float32)
b_c = tf.get_variable(name="b_c", shape=(feature_count,), initializer=initializer, dtype=tf.float32)
conv_without_bias = tf.nn.conv2d(x_image, C, strides=[1, 1, 1, 1], padding="SAME")
conv = conv_without_bias + b_c
# Branching
flat_x = tf.reshape(x, [-1, D])
W = tf.get_variable(name="W", shape=(D, k), initializer=initializer,
                    dtype=tf.float32)
b = tf.get_variable(name="b", shape=(k,), initializer=initializer, dtype=tf.float32)
activations = tf.matmul(flat_x, W) + b
h = tf.nn.softmax(logits=activations)
pass_check = tf.greater_equal(x=h, y=threshold)
eval_list = []
summation = None
for n in range(k):
    pre_mask = tf.slice(pass_check, [0, n], [-1, 1])
    mask = tf.reshape(pre_mask, [-1])
    branched_conv = tf.boolean_mask(tensor=conv, mask=mask)
    branched_activations = tf.boolean_mask(tensor=activations, mask=mask)
    flattened_conv = tf.reshape(branched_conv, [-1, D * feature_count])
    concat_list = [flattened_conv, branched_activations]
    eval_list.append(tf.concat(concat_list, axis=1))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

results = sess.run(eval_list, {x: samples})
print("X")