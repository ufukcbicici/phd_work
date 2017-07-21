import tensorflow as tf

from auxillary.constants import ProblemType, ChannelTypes, ShrinkageRegularizers, ArgumentTypes
from auxillary.tf_layer_factory import TfLayerFactory
from framework.network_argument import NetworkArgument
from framework.network_channel import NetworkChannel
from losses.l2_loss import L2Loss


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
        with NetworkChannel(node=self, channel=producer_channel, producer_node=producer_node,
                            producer_channel=producer_channel, producer_channel_index=producer_channel_index) \
                as transfer_channel:
            output_tensor = self.parentNetwork.apply_decision(node=self, channel=producer_channel,
                                                              channel_index=transfer_channel.channelIndex)
            transfer_channel.add_operation(op=output_tensor)

    def get_input(self, producer_node, channel, channel_index):
        if (producer_node, channel, channel_index) not in self.inputs:
            raise Exception("Input node found.")
        return self.inputs[(producer_node, channel, channel_index)].inputObject

    @staticmethod
    def apply_loss(loss):
        # Loss channel
        loss.build_training_network()
        # Evaluation channel
        loss.build_evaluation_network()
        # Finalize, clean up
        loss.finalize()

    def attach_loss_eval_channels(self):
        pass

    def attach_shrinkage_losses(self):
        if self.parentNetwork.shrinkageRegularizer == ShrinkageRegularizers.l2:
            for argument in self.argumentsDict.values():
                if argument.argumentType == ArgumentTypes.learnable_parameter:
                    l2_loss = L2Loss(parent_node=self, argument=argument,
                                     training_program=self.parentNetwork.parameterFile)
                    NetworkNode.apply_loss(loss=l2_loss)
        else:
            raise NotImplementedError()
