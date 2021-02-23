import numpy as np
import tensorflow as tf


class MaskedBatchNormalization(tf.keras.layers.Layer):
    def __init__(self, momentum):
        super().__init__()
        self.gamma = None
        self.beta = None
        self.popMean = None
        self.popVar = None
        self.timesCalled = None
        self.momentum = momentum

    def build(self, input_shape):
        assert len(input_shape) == 2
        param_dims = [shp[-1] for shp in input_shape]
        assert len(set(param_dims)) == 1
        input_dim = param_dims[0]
        self.gamma = self.add_weight(name="gamma",
                                     shape=input_dim,
                                     initializer=tf.keras.initializers.Constant(value=1.0),
                                     trainable=True)
        self.beta = self.add_weight(name="beta",
                                    shape=input_dim,
                                    initializer=tf.keras.initializers.Constant(value=0.0),
                                    trainable=True)
        self.popMean = self.add_weight(name="popMean",
                                       shape=input_dim,
                                       initializer=tf.keras.initializers.Constant(value=0.0),
                                       trainable=False)
        self.popVar = self.add_weight(name="popVar",
                                      shape=input_dim,
                                      initializer=tf.keras.initializers.Constant(value=1.0),
                                      trainable=False)
        self.timesCalled = self.add_weight(name="timesCalled", shape=(),
                                           initializer=tf.keras.initializers.Constant(value=0),
                                           dtype=tf.int32,
                                           trainable=False)

    def call(self, inputs, **kwargs):
        x_ = inputs[0]
        masked_x = inputs[1]
        is_training = kwargs["training"]
        mu, sigma = tf.nn.moments(masked_x, [i for i in range(len(x_.get_shape()) - 1)])
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
    batch_size = 125
    dim = 128
    momentum = 0.9
    target_val = 5.0
    x1 = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, dim))
    x2 = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, dim))
    mask_vector = np.random.randint(low=0, high=2, size=(batch_size,))

    i_x = tf.keras.Input(shape=(dim,))
    net = tf.keras.layers.Dense(256)(i_x)
    masked_net = tf.boolean_mask(net, mask_vector)
    norm_result = MaskedBatchNormalization(momentum=momentum)([net, masked_net])
    y_hat = tf.reduce_mean(norm_result, axis=-1)
    y_gt = target_val * tf.ones_like(y_hat)
    reconstruction_loss = tf.keras.losses.mean_squared_error(y_gt, y_hat)
    model = tf.keras.Model(inputs=i_x,
                           outputs={"masked_net": masked_net,
                                    "net": net,
                                    "norm_result": norm_result,
                                    "y_hat": y_hat,
                                    "y_gt": y_gt,
                                    "reconstruction_loss": reconstruction_loss})
    optimizer = tf.keras.optimizers.Adam()
    loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    for i in range(10000):
        with tf.GradientTape() as tape:
            outputs_dict = model(x1, training=True)
            loss_tracker.update_state(values=outputs_dict["reconstruction_loss"])
        grads = tape.gradient(outputs_dict["reconstruction_loss"], model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print("Loss:{0}".format(loss_tracker.result().numpy()))
    results = model(x2, training=False)
    print("x")
