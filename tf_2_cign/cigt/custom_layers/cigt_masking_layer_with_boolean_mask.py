import tensorflow as tf

from tf_2_cign.custom_layers.cign_binary_action_result_generator_layer import CignBinaryActionResultGeneratorLayer


# OK
class CigtMaskingLayerWithBooleanMask(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.inputDim = None

    def build(self, input_shape):
        self.inputDim = len(input_shape[0])

    # @tf.function
    def call(self, inputs, **kwargs):
        x_ = inputs[0]
        routing_matrix = inputs[1]
        input_shape = tf.shape(x_)
        # use_gumbel = inputs[2]
        # is_training = kwargs["training"]

        num_routes = tf.shape(routing_matrix)[-1]
        route_width = tf.shape(x_)[-1] // num_routes

        repeat_array = route_width * tf.ones_like(routing_matrix[0])
        mask_array = tf.repeat(routing_matrix, repeats=repeat_array, axis=-1)
        mask_array = tf.cast(mask_array, dtype=tf.bool)

        dim_array = tf.range(0, tf.size(input_shape))
        transpose_forward_arr = tf.concat(
            [tf.convert_to_tensor([0]), tf.expand_dims(dim_array[-1], axis=0), dim_array[1:-1]], axis=0)
        new_shape_arr = tf.concat(
            [tf.expand_dims(input_shape[0], axis=0), tf.convert_to_tensor([route_width]), input_shape[1:-1]],
            axis=0)
        transpose_back_arr = tf.concat(
            [tf.convert_to_tensor([0]), dim_array[2:], tf.convert_to_tensor([1])], axis=0)

        x_hat = tf.transpose(x_, transpose_forward_arr)
        x_hat = tf.reshape(tf.boolean_mask(x_hat, mask_array), new_shape_arr)
        x_hat = tf.transpose(x_hat, transpose_back_arr)

        return x_hat



