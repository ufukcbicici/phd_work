# import tensorflow as tf
# import numpy as np
#
# state_dim = 3
# state_count = 3
# repeat_count = 2
#
# state_input = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name="inputs")
# squared = tf.square(state_input)
# squared_and_tiled = tf.tile(squared, [repeat_count, 1])
# mean = tf.reduce_mean(squared_and_tiled)
#
# grads = tf.gradients(mean, [squared_and_tiled, squared])
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)

x = np.random.uniform(size=(state_count, state_dim))
result = sess.run([grads, mean], feed_dict={state_input: x})
print("X")