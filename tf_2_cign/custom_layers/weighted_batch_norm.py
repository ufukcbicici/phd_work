import numpy as np
import tensorflow as tf
import time

from tf_2_cign.custom_layers.masked_batch_norm import MaskedBatchNormalization
from tf_2_cign.utilities.utilities import Utilities


# tf.autograph.set_verbosity(10, True)


class WeightedBatchNormalization(tf.keras.layers.Layer):
    def __init__(self, momentum, node=None, name=""):
        super().__init__(name=name)
        self.gamma = None
        self.beta = None
        self.popMean = None
        self.popVar = None
        self.timesCalled = None
        self.momentum = momentum
        self.node = node
        self.opPrefix = name

    def build(self, input_shape):
        # assert len(input_shape) == 2
        # param_dims = [shp[-1] for shp in input_shape]
        # assert len(set(param_dims)) == 1
        input_dim = input_shape[0].as_list()[-1]

        self.gamma = self.add_weight(name=
                                     "wb_gamma" if self.node is None
                                     else Utilities.get_variable_name(name="{0}_wb_gamma".format(self.opPrefix),
                                                                      node=self.node),
                                     shape=input_dim,
                                     initializer=tf.keras.initializers.Constant(value=1.0),
                                     trainable=True)
        self.beta = self.add_weight(name="wb_beta" if self.node is None
                                    else Utilities.get_variable_name(name="{0}_wb_beta".format(self.opPrefix),
                                                                     node=self.node),
                                    shape=input_dim,
                                    initializer=tf.keras.initializers.Constant(value=0.0),
                                    trainable=True)
        self.popMean = self.add_weight(name="popMean" if self.node is None
                                       else Utilities.get_variable_name(name="{0}_popMean".format(self.opPrefix),
                                                                        node=self.node),
                                       shape=input_dim,
                                       initializer=tf.keras.initializers.Constant(value=0.0),
                                       trainable=False)
        self.popVar = self.add_weight(name="popVar" if self.node is None
                                      else Utilities.get_variable_name(name="{0}_popVar".format(self.opPrefix),
                                                                       node=self.node),
                                      shape=input_dim,
                                      initializer=tf.keras.initializers.Constant(value=1.0),
                                      trainable=False)
        self.timesCalled = self.add_weight(name="timesCalled" if self.node is None
                                           else Utilities.get_variable_name(name="timesCalled", node=self.node),
                                           shape=(),
                                           initializer=tf.keras.initializers.Constant(value=0),
                                           dtype=tf.int32,
                                           trainable=False)

    # @tf.function
    def call(self, inputs, **kwargs):
        x_ = inputs[0]
        weight_vector = inputs[1]
        is_training = kwargs["training"]
        population_count = tf.reduce_prod(tf.shape(x_)[1:-1])
        sample_count = tf.reduce_sum(weight_vector)
        probability_vector = tf.math.divide_no_nan(tf.cast(weight_vector, dtype=x_.dtype),
                                                   tf.cast(sample_count, dtype=x_.dtype))
        # probability_vector = weight_vector / tf.reduce_sum(weight_vector)
        # probability_vector = tf.cast(probability_vector, dtype=x_.dtype)
        probability_tensor = tf.ones_like(x_)
        probability_vector_expanded = tf.identity(probability_vector)
        for idx in range(len(x_.get_shape()) - 1):
            probability_vector_expanded = tf.expand_dims(probability_vector_expanded, axis=-1)
        probability_tensor = probability_vector_expanded * probability_tensor
        probability_tensor = ((1.0 / tf.cast(population_count, tf.float32)) * tf.ones_like(x_)) * probability_tensor

        # Calculate batch mean, weighted
        weighted_x = probability_tensor * x_
        mean_x = tf.reduce_sum(weighted_x, [idx for idx in range(len(x_.get_shape()) - 1)])
        # Calculate batch variance, weighted
        mean_x_expanded = tf.identity(mean_x)
        for idx in range(len(x_.get_shape()) - 1):
            mean_x_expanded = tf.expand_dims(mean_x_expanded, axis=0)
        zero_meaned = x_ - mean_x_expanded
        zero_meaned_squared = tf.square(zero_meaned)
        variance_x = tf.reduce_sum(probability_tensor * zero_meaned_squared,
                                   [idx for idx in range(len(x_.get_shape()) - 1)])
        mu = mean_x
        sigma = variance_x
        if is_training:
            final_mean = mu
            final_var = sigma
        else:
            final_mean = self.popMean
            final_var = self.popVar
        normed_x = tf.nn.batch_normalization(x=x_,
                                             mean=final_mean,
                                             variance=final_var,
                                             offset=self.beta,
                                             scale=self.gamma,
                                             variance_epsilon=1e-5)
        if is_training:
            with tf.control_dependencies([normed_x]):
                new_pop_mean = tf.where(self.timesCalled > 0,
                                        (self.momentum * self.popMean + (1.0 - self.momentum) * mu), mu)
                new_pop_var = tf.where(self.timesCalled > 0,
                                       (self.momentum * self.popVar + (1.0 - self.momentum) * sigma), sigma)
                self.timesCalled.assign_add(delta=1)
                self.popMean.assign(value=new_pop_mean)
                self.popVar.assign(value=new_pop_var)
        return normed_x


if __name__ == "__main__":
    print(tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    with tf.device("GPU"):
        batch_size = 125
        dim = 128
        momentum = 0.9
        target_val = 5.0
        # CNN Output
        x1 = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 32, 32, dim))
        x2 = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 32, 32, dim))
        i_x = tf.keras.Input(shape=(32, 32, dim))
        # Dense Output
        # x1 = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, dim))
        # x2 = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, dim))
        # i_x = tf.keras.Input(shape=(dim))

        mask_vector = np.random.randint(low=0, high=2, size=(batch_size,))

        mb_norm = MaskedBatchNormalization(momentum=momentum)
        wb_norm = WeightedBatchNormalization(momentum=momentum)

        # func_str = tf.autograph.to_code(wb_norm.call.python_function)

        mask_vector_tf = tf.keras.Input(shape=())

        # CNN Output
        net = tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=3,
                                     activation=tf.nn.relu,
                                     strides=1,
                                     padding="same",
                                     use_bias=True,
                                     name="conv")(i_x)
        # Dense Output
        # net = tf.keras.layers.Dense(256)(i_x)

        masked_net = tf.boolean_mask(net, mask_vector_tf)
        norm_result_mb = mb_norm([net, masked_net])
        norm_result_wb = wb_norm([net, mask_vector_tf])

        if len(norm_result_wb.shape) > 2:
            mb_dense = tf.keras.layers.Flatten()(norm_result_mb)
            wb_dense = tf.keras.layers.Flatten()(norm_result_wb)
        else:
            mb_dense = norm_result_mb
            wb_dense = norm_result_wb

        y_hat_mb = tf.reduce_mean(mb_dense, axis=-1)
        y_hat_wb = tf.reduce_mean(wb_dense, axis=-1)

        y_gt = target_val * tf.ones_like(y_hat_mb)
        reconstruction_loss_mb = tf.keras.losses.mean_squared_error(y_gt, y_hat_mb)
        reconstruction_loss_wb = tf.keras.losses.mean_squared_error(y_gt, y_hat_wb)
        total_loss = 0.5 * (reconstruction_loss_mb + reconstruction_loss_wb)
        model = tf.keras.Model(inputs=[i_x, mask_vector_tf],
                               outputs={"masked_net": masked_net,
                                        "net": net,
                                        "norm_result_mb": norm_result_mb,
                                        "norm_result_wb": norm_result_wb,
                                        "y_hat_mb": y_hat_mb,
                                        "y_hat_wb": y_hat_wb,
                                        "y_gt": y_gt,
                                        "total_loss": total_loss})
        optimizer = tf.keras.optimizers.Adam()
        loss_tracker = tf.keras.metrics.Mean(name="total_loss")

        for i in range(10000):
            t0 = time.time()
            with tf.GradientTape() as tape:
                outputs_dict = model([x1, mask_vector], training=True)
                if (i + 1) % 9500 == 0:
                    assert np.allclose(outputs_dict["norm_result_mb"].numpy(), outputs_dict["norm_result_wb"].numpy())
                    # Compare population means
                    assert np.allclose(model.variables[4].numpy(), model.variables[9].numpy())
                    # Compare population variances
                    assert np.allclose(model.variables[5].numpy(), model.variables[10].numpy())
                    print("All Assertions Work!!!")
                    # assert np.allclose(model.variables[4])
                loss_tracker.update_state(values=outputs_dict["total_loss"])
            grads = tape.gradient(outputs_dict["total_loss"], model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            t1 = time.time()
            print("{0} Loss:{1} Time:{2}".format(i, loss_tracker.result().numpy(), t1 - t0))

        results = model([x1, mask_vector], training=False)

        print("x")
