import numpy as np
import tensorflow as tf

from simple_tf.batch_norm import fast_tree_batch_norm

x = np.random.uniform(low=-1.0, high=1.0, size=(125, 32))
mask = np.random.binomial(n=1, p=0.5, size=(125, ))

x_input = tf.placeholder(dtype=tf.float32, shape=(125, 32))
x_mask = tf.placeholder(dtype=tf.bool, shape=(125, ))
x_masked = tf.boolean_mask(x_input, x_mask)
normed_x, normed_masked_x = fast_tree_batch_norm(x=x_input, masked_x=x_masked, )

print("X")