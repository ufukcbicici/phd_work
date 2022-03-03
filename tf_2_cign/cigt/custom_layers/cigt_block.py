import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_batch_normalization import CigtBatchNormalization
from tf_2_cign.cigt.custom_layers.cigt_masking_layer import CigtMaskingLayer
from tf_2_cign.custom_layers.cign_binary_action_result_generator_layer import CignBinaryActionResultGeneratorLayer

# OK
from tf_2_cign.custom_layers.cign_conv_layer import CignConvLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer


# class CigtBlock(tf.keras.layers.Layer):
#
#     def __init__(self, network, node, ig_activations, routing_matrix):
#         super().__init__()
#         self.network = network
#         self.node = node
#         # IMPORTANT: THESE TWO ARE ALWAYS THE OUTPUTS OF THE PREVIOUS (PARENT) BLOCK!!!
#         self.igActivations = ig_activations
#         self.routingMatrix = routing_matrix
#
#     def call(self, inputs, **kwargs):
#         pass
