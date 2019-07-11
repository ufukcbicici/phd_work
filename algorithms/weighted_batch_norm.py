import tensorflow as tf
import numpy as np

from simple_tf.uncategorized.global_params import GlobalConstants


class WeightedBatchNorm:
    @staticmethod
    def weighted_batch_norm(weights, input_tensor, is_training,
                            momentum=GlobalConstants.BATCH_NORM_DECAY, epsilon=1e-3):
        tf_x = tf.identity(input_tensor)
        # Trainable parameters
        gamma = tf.Variable(name="gamma", initial_value=tf.ones([tf_x.get_shape()[-1]]))
        beta = tf.Variable(name="beta", initial_value=tf.zeros([tf_x.get_shape()[-1]]))
        # Moving mean and variance
        pop_mean = tf.Variable(name="pop_mean", initial_value=tf.constant(0.0, shape=[tf_x.get_shape()[-1]]),
                               trainable=False)
        pop_var = tf.Variable(name="pop_variance", initial_value=tf.constant(1.0, shape=[tf_x.get_shape()[-1]]),
                              trainable=False)
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
        final_mean = tf.where(is_training > 0, mean, pop_mean)
        final_var = tf.where(is_training > 0, variance, pop_var)
        x_minus_mean = input_tensor - final_mean
        normalized_x = x_minus_mean / tf.sqrt(final_var + epsilon)
        final_x = gamma*normalized_x + beta
        # Update moving mean and variance
        with tf.control_dependencies([final_mean, final_var]):
            new_pop_mean = momentum * pop_mean + (1.0 - momentum) * final_mean
            new_pop_var = momentum * pop_var + (1.0 - momentum) * final_var
            pop_mean_assign_op = tf.assign(pop_mean, new_pop_mean)
            pop_var_assign_op = tf.assign(pop_var, new_pop_var)
            tf.add_to_collection(name=tf.GraphKeys.UPDATE_OPS, value=pop_mean_assign_op)
            tf.add_to_collection(name=tf.GraphKeys.UPDATE_OPS, value=pop_var_assign_op)
            tf_normalized_x = tf.layers.batch_normalization(inputs=tf_x,
                                                            momentum=momentum,
                                                            epsilon=epsilon,
                                                            training=True)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                final_x = tf.identity(final_x)
                tf_normalized_x = tf.identity(tf_normalized_x)
                return final_x, tf_normalized_x

        # # weighted_tensor = tf.square(tf.multiply(x_minus_mean, _p))
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     tf_normalized_x = tf.identity(tf_normalized_x)
        #     return weighted_tensor, mean, variance, normalized_x, tf_normalized_x


# Conv layer
batch_size = 250
width = 16
height = 16
channels = 64

_x = tf.placeholder(name="input", dtype=tf.float32, shape=(batch_size, width, height, channels))
_weights = tf.placeholder(name="weights", dtype=tf.float32, shape=(batch_size,))
is_train = tf.placeholder(name="is_train", dtype=tf.int32)

x = np.random.uniform(size=(batch_size, width, height, channels))
w = (1.0 / batch_size) * np.ones(shape=(batch_size,))
norm_x, tf_norm_x = WeightedBatchNorm.weighted_batch_norm(weights=_weights, input_tensor=_x, is_training=is_train)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
epoch_count = 100
vars = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if "mean" in v.name or "variance" in v.name]
for i in range(epoch_count):
    res = sess.run([norm_x, tf_norm_x], feed_dict={_x: x, _weights: w, is_train: 1})
    for v in vars:
        print(v.name)
    moments = sess.run(vars)
    print("X")
res = sess.run([norm_x, tf_norm_x], feed_dict={_x: x, _weights: w, is_train: 0})
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
