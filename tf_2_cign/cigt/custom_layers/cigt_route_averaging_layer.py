import tensorflow as tf

from tf_2_cign.custom_layers.cign_binary_action_result_generator_layer import CignBinaryActionResultGeneratorLayer


# OK
class CigtRouteAveragingLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.inputDim = None

    def build(self, input_shape):
        self.inputDim = len(input_shape[0])

    # @tf.function
    def call(self, inputs, **kwargs):
        x_ = inputs[0]
        routing_matrix = inputs[1]

        num_routes = tf.shape(routing_matrix)[-1]
        input_channel_count = tf.shape(x_)[-1]
        route_channel_count = input_channel_count // num_routes
        prev_shape = tf.shape(x_)[:-1]
        num_routes = tf.expand_dims(num_routes, axis=0)
        route_channel_count = tf.expand_dims(route_channel_count, axis=0)
        x_new_shape = tf.concat([prev_shape, num_routes, route_channel_count], axis=0)
        x_reshaped = tf.reshape(x_, x_new_shape)

        open_channels_count = tf.reduce_sum(routing_matrix, axis=1)
        x_aggregated = tf.reduce_sum(x_reshaped, axis=-2)
        coeffs_vector = tf.math.reciprocal(tf.cast(open_channels_count, dtype=tf.float32))
        for i in range(self.inputDim - 1):
            coeffs_vector = tf.expand_dims(coeffs_vector, axis=1)
        x_normalized = x_aggregated * coeffs_vector
        return x_normalized
