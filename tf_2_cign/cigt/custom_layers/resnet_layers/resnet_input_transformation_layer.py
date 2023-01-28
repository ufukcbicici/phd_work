import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_conv_batch_norm_composite_layer import CigtConvBatchNormCompositeLayer


class ResnetInputTransformationLayer(tf.keras.layers.Layer):
    def __init__(self,
                 node,
                 prev_block_path_count,
                 this_block_path_count,
                 batch_norm_type,
                 bn_momentum,
                 start_moving_averages_from_zero,
                 apply_mask_to_batch_norm,
                 first_conv_kernel_size,
                 first_conv_output_dim,
                 first_conv_stride):
        super().__init__()
        self.inputTransformationLayer = CigtConvBatchNormCompositeLayer(
            kernel_size=first_conv_kernel_size,
            num_of_filters=first_conv_output_dim,
            strides=(first_conv_stride, first_conv_stride),
            node=node,
            activation=None,
            input_path_count=prev_block_path_count,
            output_path_count=this_block_path_count,
            batch_norm_type=batch_norm_type,
            bn_momentum=bn_momentum,
            start_moving_averages_from_zero=start_moving_averages_from_zero,
            apply_mask_to_batch_norm=apply_mask_to_batch_norm,
            use_bias=False,
            padding="same")

    def call(self, inputs, **kwargs):
        f_input = inputs[0]
        routing_matrix = inputs[1]
        training = kwargs["training"]

        f_net = f_input
        f_net = self.inputTransformationLayer([f_net, routing_matrix], training=training)
        f_net = tf.nn.relu(f_net)
        return f_net
