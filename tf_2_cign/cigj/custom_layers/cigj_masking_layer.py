import tensorflow as tf

from tf_2_cign.custom_layers.cign_binary_action_result_generator_layer import CignBinaryActionResultGeneratorLayer


# OK
class CigjMaskingLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.inputDim = None

    def build(self, input_shape):
        self.inputDim = len(input_shape[0])

    # def expand_mask_array_by_two(self, x_):
    #     x_ = tf.expand_dims(x_, axis=1)
    #     x_ = tf.

    def call(self, inputs, **kwargs):
        x_ = inputs[0]
        routing_matrix = inputs[1]

        num_routes = tf.shape(routing_matrix)[-1]
        route_width = tf.shape(x_)[-1] // num_routes

        repeat_array = route_width * tf.ones_like(routing_matrix[0])
        mask_array = tf.repeat(routing_matrix, repeats=repeat_array, axis=-1)
        mask_array = tf.cast(mask_array, dtype=x_.dtype)

        for i in range(self.inputDim - 2):
            mask_array = tf.expand_dims(mask_array, axis=1)

        masked_x = mask_array * x_
        return masked_x



