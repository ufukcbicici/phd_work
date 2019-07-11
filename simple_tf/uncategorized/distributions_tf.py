import numpy as np
import tensorflow as tf

vector = tf.placeholder(name="vec", dtype=tf.float32)
sample_count = tf.shape(vector)[0]
gaussian = tf.contrib.distributions.MultivariateNormalDiag(loc=np.ones(shape=[2, ]), scale_diag=np.ones(shape=[2, ]))
scale = tf.Variable(tf.constant(value=[-2.0, 3.0], dtype=tf.float32, shape=[2, ]), name="scale")
shift = tf.Variable(tf.constant(value=[5.0, 6.0], dtype=tf.float32, shape=[2, ]), name="shift")
noise = tf.cast(gaussian.sample(sample_shape=sample_count), tf.float32)
z_noise = scale * noise + shift
loss = tf.nn.l2_loss(z_noise)
grads = tf.gradients(ys=loss, xs=[scale, shift])


sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)
res = sess.run([noise, z_noise, sample_count, grads], feed_dict={vector: np.ones(shape=(27, 5))})
print("X")