import tensorflow as tf

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants


class CustomBatchNormAlgorithms:
    BATCH_NORM_OPS = "BatchNormOps"

    @staticmethod
    def batch_norm_multi_gpu(input_tensor, is_training, momentum=GlobalConstants.BATCH_NORM_DECAY,
                             epsilon=1e-5, network=None, node=None):
        gamma_name = network.get_variable_name(node=node, name="gamma") if network is not None else "gamma"
        beta_name = network.get_variable_name(node=node, name="beta") if network is not None else "beta"
        pop_mean_name = network.get_variable_name(node=node, name="pop_mean") if network is not None else "pop_mean"
        pop_var_name = network.get_variable_name(node=node, name="pop_var") if network is not None else "pop_var"
        with tf.variable_scope("batch_norm"):
            tf_x = tf.identity(input_tensor)
            # Trainable parameters
            gamma = UtilityFuncs.create_variable(name=gamma_name,
                                                 shape=[tf_x.get_shape()[-1]],
                                                 initializer=tf.ones([tf_x.get_shape()[-1]]),
                                                 dtype=tf.float32)
            beta = UtilityFuncs.create_variable(name=beta_name,
                                                shape=[tf_x.get_shape()[-1]],
                                                initializer=tf.zeros([tf_x.get_shape()[-1]]),
                                                dtype=tf.float32)
            # Moving mean and variance
            pop_mean = UtilityFuncs.create_variable(name=pop_mean_name,
                                                    shape=[tf_x.get_shape()[-1]],
                                                    initializer=tf.constant(0.0, shape=[tf_x.get_shape()[-1]]),
                                                    dtype=tf.float32,
                                                    trainable=False)
            pop_var = UtilityFuncs.create_variable(name=pop_var_name,
                                                   shape=[tf_x.get_shape()[-1]],
                                                   initializer=tf.constant(1.0, shape=[tf_x.get_shape()[-1]]),
                                                   dtype=tf.float32,
                                                   trainable=False)
            # Calculate mean and variance
            input_dim = len(input_tensor.get_shape().as_list())
            assert input_dim == 2 or input_dim == 4
            mean = tf.reduce_mean(tf_x, axis=[ax for ax in range(input_dim - 1)])
            variance = tf.reduce_mean(tf.square(input_tensor - mean), axis=[ax for ax in range(input_dim - 1)])
            final_mean = tf.where(is_training > 0, mean, pop_mean)
            final_var = tf.where(is_training > 0, variance, pop_var)
            tf.add_to_collection(CustomBatchNormAlgorithms.BATCH_NORM_OPS, (pop_mean, final_mean))
            tf.add_to_collection(CustomBatchNormAlgorithms.BATCH_NORM_OPS, (pop_var, final_var))
            x_minus_mean = input_tensor - final_mean
            normalized_x = x_minus_mean / tf.sqrt(final_var + epsilon)
            final_x = gamma * normalized_x + beta
            # Update moving mean and variance
            with tf.control_dependencies([final_mean, final_var, final_x]):
                # return final_mean, final_var, final_x
                return final_x

    @staticmethod
    def batch_norm_multi_gpu_v2(input_tensor, is_training, momentum=GlobalConstants.BATCH_NORM_DECAY,
                                epsilon=1e-5, network=None, node=None):
        gamma_name = network.get_variable_name(node=node, name="gamma") if network is not None else "gamma"
        beta_name = network.get_variable_name(node=node, name="beta") if network is not None else "beta"
        pop_mean_name = network.get_variable_name(node=node, name="pop_mean") if network is not None else "pop_mean"
        pop_var_name = network.get_variable_name(node=node, name="pop_var") if network is not None else "pop_var"
        with tf.variable_scope("batch_norm"):
            tf_x = tf.identity(input_tensor)
            # Trainable parameters
            gamma = UtilityFuncs.create_variable(name=gamma_name,
                                                 shape=[tf_x.get_shape()[-1]],
                                                 initializer=tf.ones([tf_x.get_shape()[-1]]),
                                                 dtype=tf.float32)
            beta = UtilityFuncs.create_variable(name=beta_name,
                                                shape=[tf_x.get_shape()[-1]],
                                                initializer=tf.zeros([tf_x.get_shape()[-1]]),
                                                dtype=tf.float32)
            # Moving mean and variance
            pop_mean = UtilityFuncs.create_variable(name=pop_mean_name,
                                                    shape=[tf_x.get_shape()[-1]],
                                                    initializer=tf.constant(0.0, shape=[tf_x.get_shape()[-1]]),
                                                    dtype=tf.float32,
                                                    trainable=False)
            pop_var = UtilityFuncs.create_variable(name=pop_var_name,
                                                   shape=[tf_x.get_shape()[-1]],
                                                   initializer=tf.constant(1.0, shape=[tf_x.get_shape()[-1]]),
                                                   dtype=tf.float32,
                                                   trainable=False)
            # Calculate mean and variance
            input_dim = len(input_tensor.get_shape().as_list())
            assert input_dim == 2 or input_dim == 4
            axes = [i for i in range(input_dim - 1)]
            mu, sigma = tf.nn.moments(tf_x, axes)
            final_mean = tf.where(is_training > 0, mu, pop_mean)
            final_var = tf.where(is_training > 0, sigma, pop_var)
            tf.add_to_collection(CustomBatchNormAlgorithms.BATCH_NORM_OPS, (pop_mean, final_mean))
            tf.add_to_collection(CustomBatchNormAlgorithms.BATCH_NORM_OPS, (pop_var, final_var))
            final_x = tf.nn.batch_normalization(x=tf_x, mean=final_mean, variance=final_var, offset=beta, scale=gamma,
                                                variance_epsilon=epsilon)
        with tf.control_dependencies([final_mean, final_var, final_x]):
            # return final_mean, final_var, final_x
            return final_x

    @staticmethod
    def masked_batch_norm_multi_gpu(input_tensor, masked_input_tensor, is_training,
                                    momentum=GlobalConstants.BATCH_NORM_DECAY, epsilon=1e-5, network=None, node=None):
        gamma_name = network.get_variable_name(node=node, name="gamma") if network is not None else "gamma"
        beta_name = network.get_variable_name(node=node, name="beta") if network is not None else "beta"
        pop_mean_name = network.get_variable_name(node=node, name="pop_mean") if network is not None else "pop_mean"
        pop_var_name = network.get_variable_name(node=node, name="pop_var") if network is not None else "pop_var"
        with tf.variable_scope("batch_norm"):
            tf_x = tf.identity(input_tensor)
            tf_masked_x = tf.identity(masked_input_tensor)
            # Trainable parameters
            gamma = UtilityFuncs.create_variable(name=gamma_name,
                                                 shape=[tf_x.get_shape()[-1]],
                                                 initializer=tf.ones([tf_x.get_shape()[-1]]),
                                                 dtype=tf.float32)
            beta = UtilityFuncs.create_variable(name=beta_name,
                                                shape=[tf_x.get_shape()[-1]],
                                                initializer=tf.zeros([tf_x.get_shape()[-1]]),
                                                dtype=tf.float32)
            # Moving mean and variance
            pop_mean = UtilityFuncs.create_variable(name=pop_mean_name,
                                                    shape=[tf_x.get_shape()[-1]],
                                                    initializer=tf.constant(0.0, shape=[tf_x.get_shape()[-1]]),
                                                    dtype=tf.float32,
                                                    trainable=False)
            pop_var = UtilityFuncs.create_variable(name=pop_var_name,
                                                   shape=[tf_x.get_shape()[-1]],
                                                   initializer=tf.constant(1.0, shape=[tf_x.get_shape()[-1]]),
                                                   dtype=tf.float32,
                                                   trainable=False)
            # Calculate mean and variance
            input_dim = len(input_tensor.get_shape().as_list())
            assert input_dim == 2 or input_dim == 4
            axes = [i for i in range(input_dim - 1)]
            mu, sigma = tf.nn.moments(tf_masked_x, axes)
            final_mean = tf.where(is_training > 0, mu, pop_mean)
            final_var = tf.where(is_training > 0, sigma, pop_var)
            tf.add_to_collection(CustomBatchNormAlgorithms.BATCH_NORM_OPS, (pop_mean, final_mean))
            tf.add_to_collection(CustomBatchNormAlgorithms.BATCH_NORM_OPS, (pop_var, final_var))
            final_x = tf.nn.batch_normalization(x=tf_x, mean=final_mean, variance=final_var, offset=beta, scale=gamma,
                                                variance_epsilon=epsilon)
        with tf.control_dependencies([final_mean, final_var, final_x]):
            # return final_mean, final_var, final_x
            return final_x

    @staticmethod
    def weighted_batch_norm(weights, input_tensor, is_training,
                            momentum=GlobalConstants.BATCH_NORM_DECAY, epsilon=1e-5):
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
        final_x = gamma * normalized_x + beta
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

    # "masked_batch_norm" Nasıl çalışıyor?
    # 1) is_training_phase = 1 ise:
    # x üzerinden threshold kullanan veri, masked_x üzerinden threshold kullanmayan veri gelir.
    # masked_x üzerinden gelen veri ile ortalama(mu) ve varyans(sigma) hesaplanır.
    # Popülasyon ortalaması ve varyansı güncellenir (tf.assign komutlarıyla) -> "UPDATE_OPS" a ekle.
    # Bunları optimizer ile güncelleyeceğiz.
    # 2) is_training_phase = 0 ise: (Evaluation)
    #
    @staticmethod
    def masked_batch_norm(x, masked_x, network, node, momentum, iteration, is_training_phase):
        gamma_name = network.get_variable_name(node=node, name="gamma") if network is not None else "gamma"
        beta_name = network.get_variable_name(node=node, name="beta") if network is not None else "beta"
        pop_mean_name = network.get_variable_name(node=node, name="pop_mean") if network is not None else "pop_mean"
        pop_var_name = network.get_variable_name(node=node, name="pop_var") if network is not None else "pop_var"
        pop_mean = tf.Variable(name=pop_mean_name, initial_value=tf.constant(0.0, shape=[x.get_shape()[-1]]),
                               trainable=False)
        pop_var = tf.Variable(name=pop_var_name, initial_value=tf.constant(1.0, shape=[x.get_shape()[-1]]),
                              trainable=False)
        if GlobalConstants.USE_TRAINABLE_PARAMS_WITH_BATCH_NORM:
            gamma = tf.Variable(name=gamma_name, initial_value=tf.ones([x.get_shape()[-1]]))
            beta = tf.Variable(name=beta_name, initial_value=tf.zeros([x.get_shape()[-1]]))
            # if node is not None:
            #     node.variablesSet.add(gamma)
            #     node.variablesSet.add(beta)
        else:
            gamma = None
            beta = None
        mu, sigma = tf.nn.moments(masked_x, [0])
        final_mean = tf.where(is_training_phase > 0, mu, pop_mean)
        final_var = tf.where(is_training_phase > 0, sigma, pop_var)
        normed_x = tf.nn.batch_normalization(x=x, mean=final_mean, variance=final_var, offset=beta, scale=gamma,
                                             variance_epsilon=1e-5)
        with tf.control_dependencies([normed_x]):
            new_pop_mean = tf.where(iteration > 0, (momentum * pop_mean + (1.0 - momentum) * mu), mu)
            new_pop_var = tf.where(iteration > 0, (momentum * pop_var + (1.0 - momentum) * sigma), sigma)
            pop_mean_assign_op = tf.assign(pop_mean, new_pop_mean)
            pop_var_assign_op = tf.assign(pop_var, new_pop_var)
            tf.add_to_collection(name=tf.GraphKeys.UPDATE_OPS, value=pop_mean_assign_op)
            tf.add_to_collection(name=tf.GraphKeys.UPDATE_OPS, value=pop_var_assign_op)
            return normed_x
