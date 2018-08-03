import numpy as np
import tensorflow as tf

from simple_tf.batch_norm import fast_tree_batch_norm

x_input = tf.placeholder(dtype=tf.float32, shape=(125, 32))
x_mask = tf.placeholder(dtype=tf.bool, shape=(125,))
x_masked = tf.boolean_mask(x_input, x_mask)
iteration_holder = tf.placeholder(name="iteration", dtype=tf.int64)
is_training = tf.placeholder(name="is_training", dtype=tf.int64)

normed_x, normed_masked_x, mu, sigma = fast_tree_batch_norm(x=x_input, masked_x=x_masked, network=None, node=None,
                                                            decay=0.99, iteration=iteration_holder,
                                                            is_training_phase=is_training)

# Training
sess = tf.Session()
ma_mean = np.zeros(shape=(x_input.shape[0],))
ma_var = np.ones(shape=(x_input.shape[0],))
for iteration in range(10000):
    x = np.random.uniform(low=-1.0, high=1.0, size=(125, 32))
    mask = np.random.binomial(n=1, p=0.5, size=(125,))
    m_x = x[mask]
    mean = np.mean(m_x, axis=0)
    var = np.var(m_x, axis=0)
    feed_dict = {x_input: x, x_mask: mask, iteration_holder: iteration, is_training: 1}
    results = sess.run([normed_x, normed_masked_x, mu, sigma], feed_dict=feed_dict)
    print("X")

print("X")
