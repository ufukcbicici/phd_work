import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_conv_batch_norm_composite_layer import CigtConvBatchNormCompositeLayer
from tf_2_cign.cigt.custom_layers.cigt_identity_layer import CigtIdentityLayer


# OK
class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, in_dimension, out_dimension, node, input_path_count, output_path_count,
                 batch_norm_type, bn_momentum, start_moving_averages_from_zero,
                 apply_mask_to_batch_norm, stride=1):
        super(BasicBlock, self).__init__()
        self.inDimension = in_dimension
        self.outDimension = out_dimension
        self.node = node
        self.inputPathCount = input_path_count
        self.outputPathCount = output_path_count
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.convBnLayer1 = CigtConvBatchNormCompositeLayer(
            kernel_size=3,
            num_of_filters=out_dimension,
            strides=stride,
            node=node,
            activation=None,
            input_path_count=input_path_count,
            output_path_count=input_path_count,
            batch_norm_type=batch_norm_type,
            bn_momentum=bn_momentum,
            start_moving_averages_from_zero=start_moving_averages_from_zero,
            apply_mask_to_batch_norm=apply_mask_to_batch_norm,
            use_bias=False,
            padding="same")
        # self.relu1 = tf.keras.layers.ReLU()
        self.convBnLayer2 = CigtConvBatchNormCompositeLayer(
            kernel_size=3,
            num_of_filters=out_dimension,
            strides=1,
            node=node,
            activation=None,
            input_path_count=input_path_count,
            output_path_count=output_path_count,
            batch_norm_type=batch_norm_type,
            bn_momentum=bn_momentum,
            start_moving_averages_from_zero=start_moving_averages_from_zero,
            apply_mask_to_batch_norm=apply_mask_to_batch_norm,
            use_bias=False,
            padding="same")

        if stride != 1 or self.inDimension != self.outDimension:
            self.shortcut = CigtConvBatchNormCompositeLayer(
                kernel_size=1,
                num_of_filters=out_dimension,
                strides=stride,
                node=node,
                activation=None,
                input_path_count=input_path_count,
                output_path_count=output_path_count,
                batch_norm_type=batch_norm_type,
                bn_momentum=bn_momentum,
                start_moving_averages_from_zero=start_moving_averages_from_zero,
                apply_mask_to_batch_norm=apply_mask_to_batch_norm,
                use_bias=False,
                padding="same"
            )
        else:
            self.shortcut = CigtIdentityLayer()

    def call(self, inputs, **kwargs):
        net = inputs[0]
        routing_matrix = inputs[1]

        # Mask = 0, x < 0
        # Relu(Mask(x, 0)) = 0
        # Mask(Relu(x), 0) = 0

        # Mask = 0, x >= 0
        # Relu(Mask(x, 0)) = 0
        # Mask(Relu(x), 0) = 0

        # Mask = 1, x < 0
        # Relu(Mask(x, 1)) = 0
        # Mask(Relu(x), 1) = 0

        # Mask = 1, x >= 1
        # Relu(Mask(x, 1)) = x
        # Mask(Relu(x), 1) = x

        # First Conv - Bn pair:
        # self.bn1(self.conv1(x))
        x = self.convBnLayer1([net, routing_matrix])
        # F.relu(x)
        x = tf.nn.relu(x)

        # Second Conv - Bn pair:
        # self.bn2(self.conv2(out))
        x = self.convBnLayer2([x, routing_matrix])
        shortcut_result = self.shortcut(net)
        x_hat = shortcut_result + x
        x_hat = tf.nn.relu(x_hat)
        return x_hat
