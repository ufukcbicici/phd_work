import tensorflow as tf

from tf_2_cign.cigt.custom_layers.cigt_conv_layer import CigtConvLayer
from tf_2_cign.cigt.custom_layers.cigt_dense_layer import CigtDenseLayer
from tf_2_cign.cigt.custom_layers.resnet_layers.basic_block import BasicBlock
from tf_2_cign.custom_layers.cign_conv_layer import CignConvLayer
from tf_2_cign.custom_layers.cign_dense_layer import CignDenseLayer


class ResnetCigtLeafBlock(tf.keras.layers.Layer):
    def __init__(self,
                 node,
                 block_parameters,
                 batch_norm_type,
                 start_moving_averages_from_zero,
                 apply_mask_to_batch_norm,
                 bn_momentum,
                 prev_block_path_count,
                 this_block_path_count,
                 class_count):
        super().__init__()
        self.node = node
        self.prevBlockPathCount = prev_block_path_count
        self.thisBlockPathCount = this_block_path_count
        self.blockParameters = block_parameters
        self.batchNormType = batch_norm_type
        self.startMovingAveragesFromZero = start_moving_averages_from_zero
        self.applyMaskToBatchNorm = apply_mask_to_batch_norm
        self.bnMomentum = bn_momentum
        self.blockList = []

        # block_params_object is a dictionary with required parameters stored in key-value format.
        for block_params_object in self.blockParameters:
            # Number of feature maps entering the block
            in_dimension = block_params_object["in_dimension"]
            # Number of feature maps exiting the block
            out_dimension = block_params_object["out_dimension"]
            # Number of routes entering the block
            input_path_count = block_params_object["input_path_count"]
            # Number of routes exiting the block
            output_path_count = block_params_object["output_path_count"]
            # Stride of the block's input convolution layer. When this is larger than 1, it means that we are going to
            # apply dimension reduction to feature maps.
            stride = block_params_object["stride"]

            block = BasicBlock(in_dimension=in_dimension,
                               out_dimension=out_dimension,
                               node=self.node,
                               input_path_count=input_path_count,
                               output_path_count=output_path_count,
                               batch_norm_type=self.batchNormType,
                               bn_momentum=self.bnMomentum,
                               start_moving_averages_from_zero=self.startMovingAveragesFromZero,
                               apply_mask_to_batch_norm=self.applyMaskToBatchNorm,
                               stride=stride)
            self.blockList.append(block)
        self.avgPoolingLayer = tf.keras.layers.AveragePooling2D(8)
        self.flattenLayer = tf.keras.layers.Flatten()
        self.lossLayer = CignDenseLayer(output_dim=class_count,
                                        activation=None,
                                        node=node,
                                        use_bias=True,
                                        name="loss_layer")
        # Amend the cost for routing
        cost = self.node.opMacCostsDict[self.lossLayer.opName]
        self.node.macCost -= cost
        routed_cost = cost / self.thisBlockPathCount
        self.node.opMacCostsDict[self.lossLayer.opName] = routed_cost
        self.node.macCost += routed_cost

    def call(self, inputs, **kwargs):
        f_input = inputs[0]
        # ig_activations_parent = inputs[1]
        routing_matrix = inputs[1]
        labels = tf.cast(inputs[2], dtype=tf.int32)
        training = kwargs["training"]

        # F ops
        f_net = f_input
        for block in self.blockList:
            f_net = block([f_net, routing_matrix], training=training)

        # Loss layer
        logits = self.lossLayer(f_net)
        posteriors = tf.nn.softmax(logits)
        cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        classification_loss = tf.reduce_mean(cross_entropy_loss_tensor)

        return logits, posteriors, classification_loss


    # # @tf.function
    # def call(self, inputs, **kwargs):
    #     f_input = inputs[0]
    #     # ig_activations_parent = inputs[1]
    #     routing_matrix = inputs[1]
    #     labels = tf.cast(inputs[2], dtype=tf.int32)
    #     training = kwargs["training"]
    #
    #     # F ops -  # 1 Conv layer
    #     if self.convLayer is not None and self.maxPoolLayer is not None:
    #         f_net = self.convLayer([f_input, routing_matrix])
    #         f_net = self.maxPoolLayer(f_net)
    #     else:
    #         f_net = tf.identity(f_input)
    #
    #     # F ops - Dense layers
    #     f_net = self.flattenLayer(f_net)
    #     for layer_id in range(len(self.hiddenLayers)):
    #         f_net = self.hiddenLayers[layer_id]([f_net, routing_matrix])
    #         f_net = self.dropoutLayers[layer_id](f_net)
    #
    #     # Loss layer
    #     logits = self.lossLayer(f_net)
    #     posteriors = tf.nn.softmax(logits)
    #     cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    #     classification_loss = tf.reduce_mean(cross_entropy_loss_tensor)
    #
    #     return logits, posteriors, classification_loss
    #
    #
    #
    #
    #
