import tensorflow as tf
from auxillary.constants import ArgumentTypes, ProblemType, OperationTypes, InitType, ActivationType, InputNames
from auxillary.tf_layer_factory import TfLayerFactory
from framework.network_argument import NetworkArgument
from framework.network_channel import NetworkChannel


class NetworkInput:
    def __init__(self, source_node, source_channel, source_channel_index,
                 producer_node, producer_channel, producer_channel_index,
                 input_object):
        self.inputObject = input_object
        self.sourceNode = source_node
        self.sourceChannel = source_channel
        self.sourceChannelIndex = source_channel_index
        # This triple indicates which node actually produces that output. All other nodes in between only applies
        # decision transformations on that output.
        self.producerNode = producer_node
        self.producerChannel = producer_channel
        self.producerChannelIndex = producer_channel_index


class NetworkOutput:
    def __init__(self, node, channel, channel_index,
                 producer_node, producer_channel, producer_channel_index,
                 output_object):
        self.outputObject = output_object
        # This triple indicates directly from which channel and channel index this output goes through.
        self.currentNode = node
        self.currentChannel = channel
        self.currentChannelIndex = channel_index
        # This triple indicates to which node, which channel and which channel index this output belongs.
        self.producerNode = producer_node
        self.producerChannel = producer_channel
        self.producerChannelIndex = producer_channel_index

    def produce_input(self):
        input_object = NetworkInput(source_node=self.currentNode.currentNode,
                                    source_channel=self.currentNode.currentChannel,
                                    source_channel_index=self.currentNode.currentChannelIndex,
                                    input_object=self.outputObject,
                                    producer_node=self.producerNode,
                                    producer_channel=self.producerChannel,
                                    producer_channel_index=self.producerChannelIndex)
        return input_object


class NetworkNode:
    def __init__(self, index, containing_network, is_root, is_leaf):
        self.index = index
        self.indicatorText = "Node_{0}".format(self.index)
        self.argumentsDict = {}
        self.parentNetwork = containing_network
        self.isRoot = is_root
        self.isLeaf = is_leaf
        self.networkChannels = {}
        self.inputs = {}
        self.outputs = {}

    # Tensorflow specific code (This should be isolated at some point in future)
    def create_variable(self, name, shape, initializer, needs_gradient, dtype, arg_type, channel):
        argument_name = "{0}_{1}".format(self.indicatorText, name)
        if argument_name in self.argumentsDict:
            raise Exception("Another argument with name {0} exists.".format(argument_name))
        variable_object = tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=dtype)
        argument = NetworkArgument(name=argument_name, symbolic_network_object=variable_object, container_node=self,
                                   needs_gradient=needs_gradient,
                                   arg_type=arg_type)
        self.argumentsDict[argument_name] = argument
        channel.add_operation(op=variable_object)
        return variable_object

    def create_transfer_channel(self, producer_node, producer_channel, producer_channel_index):
        with tf.variable_scope(self.indicatorText):
            with NetworkChannel(channel_name=producer_channel, node=self,
                                producer_node=producer_node, producer_channel=producer_channel,
                                producer_channel_index=producer_channel_index) as transfer_channel:
                output_tensor = self.parentNetwork.apply_decision(node=self, channel=producer_channel,
                                                                  channel_index=transfer_channel.channelIndex)
                transfer_channel.add_operation(op=output_tensor)

    def attach_loss(self):
        raise NotImplementedError()
        # if self.parentNetwork.problemType == ProblemType.classification:
        #     # Get f, h and ancestor channels, concatenate the outputs
        #     # Shapes are constrained to be 2 dimensional. Else, it will raise exception. We have to flatten all tensors
        #     # before the loss operation.
        #     tensor_list = []
        #     relevant_channels = {OperationTypes.f_operator.value, OperationTypes.h_operator.value,
        #                          OperationTypes.ancestor_activation.value}
        #     for channel_name in relevant_channels:
        #         channel_of_type = tf.get_collection(key=channel_name, scope=self.indicatorText)
        #         if len(channel_of_type) == 0:
        #             continue
        #         output_tensor = channel_of_type[-1]
        #         if len(output_tensor.shape) != 2:
        #             raise Exception("Tensors entering the loss must be 2D.")
        #         if output_tensor.shape[1].value is None:
        #             raise Exception("Output tensor's dim1 cannot be None.")
        #         tensor_list.append(output_tensor)
        #     # Pre-Loss channel
        #     class_count = self.parentNetwork.dataset.get_label_count()
        #     with tf.variable_scope(self.indicatorText):
        #         with NetworkChannel(channel_name=OperationTypes.pre_loss.value, node=self) as loss_channel:
        #             final_feature = loss_channel.add_operation(
        #                 op=tf.concat(values=tensor_list, axis=1, name="concatLoss"))
        #             final_dimension = final_feature.shape[1].value
        #             logits = TfLayerFactory.create_fc_layer(node=self, channel=loss_channel,
        #                                                     input_tensor=final_feature,
        #                                                     fc_shape=[final_dimension, class_count],
        #                                                     init_type=self.parentNetwork.lossLayerInit,
        #                                                     activation_type=self.parentNetwork.lossLayerActivation,
        #                                                     post_fix="loss")
        #             # Get the label tensor
        #             input_channel = tf.get_collection(key=OperationTypes.input.value, scope=self.indicatorText)
        #             label_tensor = None
        #             for input_tensor in input_channel:
        #                 if InputNames.label_input.value in input_tensor.name:
        #                     label_tensor = input_tensor
        #                     break
        #             if label_tensor is None:
        #                 raise Exception("No label input in Node {0}".format(self.index))
        #             # Apply softmax
        #             softmax = loss_channel.add_operation(
        #                 op=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_tensor, logits=logits,
        #                                                                   name="softmax"))
        #             cross_entropy = loss_channel.add_operation(op=tf.reduce_mean(softmax, name="cross_entropy"))
        # else:
        #     raise NotImplementedError()
