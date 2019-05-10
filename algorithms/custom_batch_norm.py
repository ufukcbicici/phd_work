import tensorflow as tf

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants


class CustomBatchNorm:
    BATCH_NORM_OPS = "MultiGPUBatchNormOps"

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
            tf.add_to_collection(CustomBatchNorm.BATCH_NORM_OPS, (pop_mean, final_mean))
            tf.add_to_collection(CustomBatchNorm.BATCH_NORM_OPS, (pop_var, final_var))
            x_minus_mean = input_tensor - final_mean
            normalized_x = x_minus_mean / tf.sqrt(final_var + epsilon)
            final_x = gamma * normalized_x + beta
            # Update moving mean and variance
            with tf.control_dependencies([final_mean, final_var, final_x]):
                # return final_mean, final_var, final_x
                return final_x

