import tensorflow as tf
import numpy as np

from simple_tf.global_params import GlobalConstants


class WeightedBatchNorm:
    @staticmethod
    def weighted_batch_norm(weights, input_tensor, momentum=GlobalConstants.BATCH_NORM_DECAY, epsilon=1e-3):
        tf_x = tf.identity(input_tensor)
        # Normalize weights
        assert len(weights.get_shape().as_list()) == 1
        sum_weights = tf.reduce_sum(weights)
        # _p = tf.expand_dims(weights / sum_weights, axis=-1)
        input_dim = len(input_tensor.get_shape().as_list())
        assert input_dim == 2 or input_dim == 4
        _p = weights / sum_weights
        if input_dim == 4:
            _p = _p / (input_tensor.get_shape().as_list()[1] * input_tensor.get_shape().as_list()[2])
        for _ in range(input_dim - 1):
            _p = tf.expand_dims(_p, axis=-1)
        weighted_tensor = tf.multiply(input_tensor, _p)
        mean = tf.reduce_sum(weighted_tensor, axis=[ax for ax in range(input_dim - 1)])
        variance = tf.reduce_sum(tf.multiply(tf.square(input_tensor - mean), _p),
                                 axis=[ax for ax in range(input_dim - 1)])
        x_minus_mean = input_tensor - mean
        normalized_x = x_minus_mean / tf.sqrt(variance + epsilon)
        tf_normalized_x = tf.layers.batch_normalization(inputs=tf_x,
                                                        momentum=0.0,
                                                        epsilon=epsilon,
                                                        training=True)
        # weighted_tensor = tf.square(tf.multiply(x_minus_mean, _p))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            tf_normalized_x = tf.identity(tf_normalized_x)
            return weighted_tensor, mean, variance, normalized_x, tf_normalized_x


# Conv layer
batch_size = 250
width = 16
height = 16
channels = 64

_x = tf.placeholder(name="input", dtype=tf.float32, shape=(batch_size, width, height, channels))
_weights = tf.placeholder(name="weights", dtype=tf.float32, shape=(batch_size,))

x = np.random.uniform(size=(batch_size, width, height, channels))
w = (1.0 / batch_size) * np.ones(shape=(batch_size,))
wt, mu, var, norm_x, tf_norm_x = WeightedBatchNorm.weighted_batch_norm(weights=_weights, input_tensor=_x)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
res = sess.run([wt, mu, var, norm_x, tf_norm_x], feed_dict={_x: x, _weights: w})
vars = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if "mean" in v.name or "variance" in v.name]
for v in vars:
    print(v.name)
moments = sess.run(vars)
print("X")


# FC Layer
# batch_size = 250
# # width = 16
# # height = 16
# channels = 64
#
# _x = tf.placeholder(name="input", dtype=tf.float32, shape=(batch_size, channels))
# _weights = tf.placeholder(name="weights", dtype=tf.float32, shape=(batch_size,))
#
# x = np.random.uniform(size=(batch_size, channels))
# w = (1.0 / batch_size) * np.ones(shape=(batch_size,))
# wt, mu, var, norm_x, tf_norm_x = WeightedBatchNorm.weighted_batch_norm(weights=_weights, input_tensor=_x)
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# res = sess.run([wt, mu, var, norm_x, tf_norm_x], feed_dict={_x: x, _weights: w})
# print("X")