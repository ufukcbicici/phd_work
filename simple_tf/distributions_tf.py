import numpy as np
import tensorflow as tf

vector = tf.placeholder(name="vec", dtype=tf.float32)
sample_count = tf.size(vector)
gaussian = tf.contrib.distributions.MultivariateNormalDiag(loc=np.ones(shape=[2, ]), scale_diag=np.ones(shape=[2, ]))
sample = gaussian.sample(sample_shape=sample_count)

sess = tf.Session()
init = tf.global_variables_initializer()

res = sess.run(sample, feed_dict={vector: np.ones(shape=(15, ))})
print("X")