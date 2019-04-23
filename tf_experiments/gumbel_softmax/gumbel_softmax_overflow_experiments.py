import tensorflow as tf
import numpy as np

from simple_tf.info_gain import InfoGainLoss

logits = tf.placeholder(name="probs", dtype=tf.float32, shape=(1, 3))
labels = tf.placeholder(name="labels", dtype=tf.float32, shape=(1, 10))
probs = tf.nn.softmax(logits)
ig = InfoGainLoss.get_loss(p_n_given_x_2d=probs, p_c_given_x_2d=labels, balance_coefficient=1.0)
grads = tf.gradients(ig, logits)
sess = tf.Session()

x = np.array([-38.396988, -19.412645, 51.50944]).reshape((1, 3))
l = np.zeros(10)
l[3] = 1.0
l = l.reshape((1, 10))
results = sess.run([ig, probs, grads], feed_dict={logits: x, labels: l})
print("X")
