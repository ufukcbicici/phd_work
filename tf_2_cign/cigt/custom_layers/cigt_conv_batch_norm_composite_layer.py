import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_batch_normalization import CigtBatchNormalization
from tf_2_cign.cigt.custom_layers.cigt_conv_layer import CigtConvLayer
from tf_2_cign.cigt.custom_layers.cigt_masking_layer import CigtMaskingLayer
from tf_2_cign.cigt.custom_layers.cigt_probabilistic_batch_normalization import CigtProbabilisticBatchNormalization
from tf_2_cign.cigt.custom_layers.cigt_standard_batch_normalization import CigtStandardBatchNormalization
from tf_2_cign.custom_layers.cign_conv_layer import CignConvLayer
from tf_2_cign.utilities.utilities import Utilities


# OK
class CigtConvBatchNormCompositeLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size, num_of_filters, strides, node, activation,
                 input_path_count,
                 output_path_count,
                 batch_norm_type,
                 bn_momentum,
                 start_moving_averages_from_zero,
                 apply_mask_to_batch_norm,
                 use_bias=True,
                 padding="same",
                 name="conv_batch_norm_composite"):
        super().__init__(name=name)
        self.batchNormType = batch_norm_type
        assert batch_norm_type in {"StandardBatchNormalization",
                                   "CigtBatchNormalization",
                                   "CigtProbabilisticBatchNormalization"}
        assert isinstance(strides, tuple)
        self.node = node
        self.batchNormMomentum = bn_momentum
        self.startMovingAverageFromZero = start_moving_averages_from_zero
        self.applyMaskToBatchNorm = apply_mask_to_batch_norm
        self.conv = CigtConvLayer(kernel_size=kernel_size, num_of_filters=num_of_filters,
                                  strides=strides, node=node, activation=activation,
                                  input_path_count=input_path_count, output_path_count=output_path_count,
                                  use_bias=use_bias, padding=padding, name="{0}_conv_op".format(self.name))
        if batch_norm_type == "StandardBatchNormalization":
            self.batchNorm = CigtStandardBatchNormalization(momentum=self.batchNormMomentum,
                                                            epsilon=1e-3,
                                                            node=self.node,
                                                            name="{0}_bn_op".format(self.name))
        elif batch_norm_type == "CigtBatchNormalization":
            self.batchNorm = CigtBatchNormalization(momentum=self.batchNormMomentum,
                                                    epsilon=1e-3,
                                                    start_moving_averages_from_zero=self.startMovingAverageFromZero,
                                                    node=self.node,
                                                    name="{0}_bn_op".format(self.name))
        elif batch_norm_type == "CigtProbabilisticBatchNormalization":
            self.batchNorm = CigtProbabilisticBatchNormalization(
                momentum=self.batchNormMomentum,
                epsilon=1e-3,
                start_moving_averages_from_zero=self.startMovingAverageFromZero,
                node=self.node,
                normalize_routing_matrix=True,
                name="{0}_bn_op".format(self.name))
        else:
            raise ValueError("Unknown batch normalization type:{0}".format(batch_norm_type))
        self.cigtMaskingLayer = CigtMaskingLayer()

    def call(self, inputs, **kwargs):
        net = inputs[0]
        routing_matrix = inputs[1]
        training = kwargs["training"]

        # F ops
        x = self.conv([net, routing_matrix])
        x = self.batchNorm([x, routing_matrix], training=training)
        if self.applyMaskToBatchNorm:
            x = self.cigtMaskingLayer([x, routing_matrix])
        return x
