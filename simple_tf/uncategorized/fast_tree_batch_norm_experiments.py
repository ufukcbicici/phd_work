import numpy as np
import tensorflow as tf

from simple_tf.uncategorized.batch_norm import fast_tree_batch_norm

decay = 0.99
x_input = tf.placeholder(dtype=tf.float32, shape=(125, 32))
x_mask = tf.placeholder(dtype=tf.bool, shape=(125,))
x_masked = tf.boolean_mask(x_input, x_mask)
iteration_holder = tf.placeholder(name="iteration", dtype=tf.int64)
is_training = tf.placeholder(name="is_training", dtype=tf.int64)
population_mean = tf.Variable(name="population_mean", initial_value=tf.constant(0.0), trainable=False)
population_var = tf.Variable(name="population_var", initial_value=tf.constant(0.0), trainable=False)

normed_x, normed_masked_x, mu, sigma, final_mean, final_var, pop_mean, pop_var = fast_tree_batch_norm(x=x_input,
                                                                                                      masked_x=x_masked,
                                                                                                      network=None,
                                                                                                      node=None,
                                                                                                      decay=decay,
                                                                                                      iteration=iteration_holder,
                                                                                                      is_training_phase=is_training)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    foo_op = mu + sigma

# Training
sess = tf.Session()
ma_mean = np.zeros(shape=(x_input.shape[0],))
ma_var = np.ones(shape=(x_input.shape[0],))
sess.run(tf.global_variables_initializer())
for iteration in range(10000):
    # Simulate batch normalization manually for training
    x = np.random.uniform(low=-1.0, high=1.0, size=(125, 32))
    mask = np.random.binomial(n=1, p=0.5, size=(125,))
    non_zero_indices = np.nonzero(mask)
    m_x = x[non_zero_indices]
    mean = np.mean(m_x, axis=0)
    var = np.var(m_x, axis=0)
    normalized_x = (x - mean) / np.sqrt(var + 1e-5)
    normalized_mask_x = (m_x - mean) / np.sqrt(var + 1e-5)
    ma_mean = (decay * ma_mean + (1.0 - decay) * mean) if iteration > 0 else mean
    ma_var = (decay * ma_var + (1.0 - decay) * var) if iteration > 0 else var
    # Simulate batch normalization manually for training

    feed_dict = {x_input: x, x_mask: mask, iteration_holder: iteration, is_training: 1}
    results = sess.run([normed_x, normed_masked_x, mu, sigma, final_mean, final_var, foo_op], feed_dict=feed_dict)
    population_stats = sess.run([pop_mean, pop_var])
    print("normalized_x == normed_x:{0}".format(np.allclose(normalized_x, results[0], rtol=1.e-3, atol=1.e-6)))
    assert np.allclose(normalized_x, results[0], rtol=1.e-3, atol=1.e-6)
    print("normalized_mask_x == normed_masked_x:{0}".format(
        np.allclose(normalized_mask_x, results[1], rtol=1.e-3, atol=1.e-6)))
    assert np.allclose(normalized_mask_x, results[1], rtol=1.e-3, atol=1.e-6)
    print("mean == mu:{0}".format(np.allclose(mean, results[2], rtol=1.e-3, atol=1.e-6)))
    assert np.allclose(mean, results[2], rtol=1.e-3, atol=1.e-6)
    print("var == sigma:{0}".format(np.allclose(var, results[3], rtol=1.e-3, atol=1.e-6)))
    assert np.allclose(var, results[3], rtol=1.e-3, atol=1.e-6)
    print("ma_mean == pop_mean:{0}".format(np.allclose(ma_mean, population_stats[0], rtol=1.e-3, atol=1.e-6)))
    assert np.allclose(ma_mean, population_stats[0], rtol=1.e-3, atol=1.e-6)
    print("ma_var == pop_var:{0}".format(np.allclose(ma_var, population_stats[1], rtol=1.e-3, atol=1.e-6)))
    assert np.allclose(ma_var, population_stats[1], rtol=1.e-3, atol=1.e-6)
    print("results[0][non_zero_indices] == results[1]:{0}".
          format(np.allclose(results[0][non_zero_indices], results[1])))
    assert np.allclose(results[0][non_zero_indices], results[1])
    print("Iteration:{0}".format(iteration))

# Testing
for iteration in range(10000):
    # Simulate batch normalization manually for testing
    x = np.random.uniform(low=-1.0, high=1.0, size=(125, 32))
    mask = np.random.binomial(n=1, p=0.5, size=(125,))
    non_zero_indices = np.nonzero(mask)
    m_x = x[non_zero_indices]
    ones_mask = np.ones(shape=(m_x.shape[0], ), dtype=np.int64)
    normalized_x = (x - ma_mean) / np.sqrt(ma_var + 1e-5)
    normalized_mask_x = (m_x - ma_mean) / np.sqrt(ma_var + 1e-5)
    # Simulate batch normalization manually for testing
    feed_dict = {x_input: x, x_mask: mask, iteration_holder: iteration, is_training: 0}
    results = sess.run([normed_x, normed_masked_x, mu, sigma, final_mean, final_var], feed_dict=feed_dict)
    print("normalized_x == normed_x:{0}".format(np.allclose(normalized_x, results[0], rtol=1.e-3, atol=1.e-6)))
    assert np.allclose(normalized_x, results[0], rtol=1.e-3, atol=1.e-6)
    print("normed_masked_x == normed_masked_x:{0}".format(
        np.allclose(normalized_mask_x, results[1], rtol=1.e-3, atol=1.e-6)))
    assert np.allclose(normalized_mask_x, results[1], rtol=1.e-3, atol=1.e-6)
    print("Iteration:{0}".format(iteration))
print("X")
