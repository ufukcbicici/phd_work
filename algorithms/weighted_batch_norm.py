import tensorflow as tf
import numpy as np

from simple_tf.global_params import GlobalConstants


class WeightedBatchNorm:
    @staticmethod
    def weighted_batch_norm(weights, input_tensor, momentum=GlobalConstants.BATCH_NORM_DECAY, epsilon=1e-3):
        # Normalize weights
        assert len(weights.get_shape().as_list()) == 1
        sum_weights = tf.reduce_sum(weights)
        # _p = tf.expand_dims(weights / sum_weights, axis=-1)
        _p = weights / sum_weights
        input_dim = len(input_tensor.get_shape().as_list())
        assert input_dim == 2 or input_dim == 4
        for _ in range(input_dim - 1):
            _p = tf.expand_dims(_p, axis=-1)
        weighted_tensor = tf.multiply(input_tensor, _p)
        mean = tf.reduce_sum(weighted_tensor, axis=[ax for ax in range(input_dim - 1)])
        x_minus_mean = input_tensor - mean
        return weighted_tensor, mean, x_minus_mean


batch_size = 250
width = 16
height = 16
channels = 64

_x = tf.placeholder(name="input", dtype=tf.float32, shape=(batch_size, width, height, channels))
_weights = tf.placeholder(name="weights", dtype=tf.float32, shape=(batch_size,))

x = np.random.uniform(size=(batch_size, width, height, channels))
w = (1.0 / batch_size) * np.ones(shape=(batch_size,))
wt, mu, x_min_mu = WeightedBatchNorm.weighted_batch_norm(weights=_weights, input_tensor=_x)

sess = tf.Session()
res = sess.run([wt, mu, x_min_mu], feed_dict={_x: x, _weights: w})
print("X")
