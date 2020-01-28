import numpy as np
import tensorflow as tf


epsilon = 1e-30
policy_count = 3
state_count = 10

policy = tf.placeholder(dtype=tf.float32, shape=[None, policy_count])
log_policy = tf.log(policy)
inf_mask = tf.greater_equal(policy, 1.0 - epsilon)
inf_detection_vector = tf.reduce_any(inf_mask, axis=1)


policy_simulated = np.random.uniform(size=(state_count, policy_count))
policy_simulated[2, :] = np.array([0.1, 60, 0.1])
policy_simulated = np.exp(policy_simulated) / np.expand_dims(np.sum(np.exp(policy_simulated), axis=1), axis=1)

sess = tf.Session()
results = sess.run([policy, log_policy], feed_dict={policy: policy_simulated})
print("X")

