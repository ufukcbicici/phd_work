import tensorflow as tf
import numpy as np


# probs = tf.placeholder(name="probs", dtype=tf.float32)
# temperature = tf.placeholder(name="temperature", dtype=tf.float32)
# dist = tf.contrib.distributions.RelaxedOneHotCategorical(temperature=temperature, probs=probs)
#
# prob_vector = np.array([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.5, 0.2]])
# reparam_type = dist.reparameterization_type
# samples = dist.sample()
# grads = tf.gradients(samples, probs)
#
# sess = tf.Session()
# results = sess.run([samples, grads], feed_dict={temperature: 0.5, probs: prob_vector})
# print("X")

probs = tf.placeholder(name="probs", dtype=tf.float32)
temperature = tf.placeholder(name="temperature", dtype=tf.float32)
prob_vector = np.array([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.5, 0.2]])
dist = tf.contrib.distributions.RelaxedOneHotCategorical(temperature=temperature, probs=probs)
samples = dist.sample()
arg_max_samples = tf.argmax(samples, axis=1)
one_hot_samples = tf.one_hot(indices=arg_max_samples, depth=4, axis=-1, dtype=tf.int64)
sum = tf.reduce_sum(one_hot_samples)
grads = tf.gradients(sum, one_hot_samples)

sess = tf.Session()
results = sess.run([samples, sum, grads, arg_max_samples, one_hot_samples], feed_dict={temperature: 0.5, probs: prob_vector})
print("X")