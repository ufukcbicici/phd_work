import tensorflow as tf

from tf_2_cign.cigj.custom_layers.cigj_masking_layer import CigjMaskingLayer
from tf_2_cign.custom_layers.cign_binary_action_result_generator_layer import CignBinaryActionResultGeneratorLayer


# OK
from tf_2_cign.custom_layers.cign_conv_layer import CignConvLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer


class CigjBlock(tf.keras.layers.Layer):

    def __init__(self, network, ig_activations, routing_matrix):
        super().__init__()
        self.network = network
        self.igActivations = ig_activations
        self.routingMatrix = routing_matrix
        self.operationLayers = []
        self.maskingLayers = []

    # def create_masking_layers(self):
    #     for op in self.operationLayers:
    #         if isinstance(op, CignConvLayer) or isinstance(op, CignDenseLayer):
    #             masking_layer = CigjMaskingLayer()
    #             self.maskingLayers.append(masking_layer)
    #         else:
    #             self.maskingLayers.append(None)

    def block_function(self):
        pass

    def mask_layer(self, net, mask_op):
        assert isinstance(mask_op, CigjMaskingLayer)
        masked_net = mask_op([net, self.routingMatrix])
        return masked_net

    def call(self, inputs, **kwargs):
        pass
