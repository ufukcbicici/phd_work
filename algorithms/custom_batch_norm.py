import numpy as np
import tensorflow as tf

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants


class CustomBatchNorm:
    @staticmethod
    def batch_norm(input_tensor, is_training, momentum=GlobalConstants.BATCH_NORM_DECAY, epsilon=1e-3):
        with tf.name_scope("custom_batch_norm"):
            tf_x = tf.identity(input_tensor)
            # Trainable parameters
            gamma = UtilityFuncs.create_variable(name="gamma",
                                                 shape=[tf_x.get_shape()[-1]],
                                                 initializer=tf.ones([tf_x.get_shape()[-1]]),
                                                 type=tf.float32)
            beta = UtilityFuncs.create_variable(name="beta",
                                                shape=[tf_x.get_shape()[-1]],
                                                initializer=tf.zeros([tf_x.get_shape()[-1]]),
                                                type=tf.float32)
            # Moving mean and variance
            pop_mean = UtilityFuncs.create_variable(name="pop_mean",
                                                    shape=[tf_x.get_shape()[-1]],
                                                    initializer=tf.constant(0.0, shape=[tf_x.get_shape()[-1]]),
                                                    type=tf.float32,
                                                    trainable=False)
            pop_var = UtilityFuncs.create_variable(name="pop_var",
                                                   shape=[tf_x.get_shape()[-1]],
                                                   initializer=tf.constant(1.0, shape=[tf_x.get_shape()[-1]]),
                                                   type=tf.float32,
                                                   trainable=False)
            # Calculate mean and variance
            input_dim = len(input_tensor.get_shape().as_list())
            assert input_dim == 2 or input_dim == 4
            mean = tf.reduce_mean(tf_x, axis=[ax for ax in range(input_dim - 1)])
            variance = tf.reduce_mean(tf.square(input_tensor - mean), axis=[ax for ax in range(input_dim - 1)])
            final_mean = tf.where(is_training > 0, mean, pop_mean)
            final_var = tf.where(is_training > 0, variance, pop_var)
            x_minus_mean = input_tensor - final_mean
            normalized_x = x_minus_mean / tf.sqrt(final_var + epsilon)
            final_x = gamma * normalized_x + beta
            # Update moving mean and variance
            with tf.control_dependencies([final_mean, final_var, final_x]):
                return final_mean, final_var, final_x


# Conv layer
batch_size = 250
width = 16
height = 16
channels = 64
_x = tf.placeholder(name="input", dtype=tf.float32, shape=(batch_size, width, height, channels))
is_train = tf.placeholder(name="is_train", dtype=tf.int32)

mu, sigma, normalized_x = CustomBatchNorm.batch_norm(input_tensor=_x,
                                                     momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                     epsilon=1e-3,
                                                     is_training=is_train)
tf_normalized_x = tf.layers.batch_normalization(inputs=_x,
                                                momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                epsilon=1e-3,
                                                training=tf.cast(is_train, tf.bool))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

x = np.random.uniform(size=(batch_size, width, height, channels))

res = sess.run([mu, sigma, normalized_x, tf_normalized_x], feed_dict={_x: x, is_train: 1})
print("X")
