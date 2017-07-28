import tensorflow as tf

from auxillary.constants import ProblemType, ChannelTypes, ShrinkageRegularizers, ArgumentTypes
from auxillary.tf_layer_factory import TfLayerFactory
from framework.network_argument import NetworkArgument
from framework.network_channel import NetworkChannel
from losses.l2_loss import L2Loss


class NetworkNode:
    def __init__(self, index, containing_network, is_root, is_leaf, is_accumulation):
        self.index = index
        self.indicatorText = "Node_{0}".format(self.index)
        self.argumentsDict = {}
        self.parentNetwork = containing_network
        self.isRoot = is_root
        self.isLeaf = is_leaf
        self.isAccumulation = is_accumulation
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

    def add_io_object(self, producer_node, channel, channel_index, object, container):
        if (producer_node, channel, channel_index) in container:
            raise Exception("Input already exists.")
        container[(producer_node, channel, channel_index)] = object

    def add_input(self, producer_triple, input_object):
        self.add_io_object(producer_node=producer_triple[0], channel=producer_triple[1],
                           channel_index=producer_triple[2], object=input_object, container=self.inputs)

    def add_output(self, producer_triple, output_object):
        self.add_io_object(producer_node=producer_triple[0], channel=producer_triple[1],
                           channel_index=producer_triple[2], object=output_object, container=self.outputs)

    def get_input(self, producer_triple):
        producer_node = producer_triple[0]
        channel = producer_triple[1]
        channel_index = producer_triple[2]
        if (producer_node, channel, channel_index) not in self.inputs:
            raise Exception("Input node found.")
        return self.inputs[(producer_node, channel, channel_index)]

    def get_output(self, producer_triple):
        producer_node = producer_triple[0]
        channel = producer_triple[1]
        channel_index = producer_triple[2]
        if (producer_node, channel, channel_index) not in self.outputs:
            raise Exception("Output node found.")
        return self.outputs[(producer_node, channel, channel_index)]

    # Methods to override
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

    def attach_decision(self):
        pass

    def apply_decision(self, tensor):
        pass

    @staticmethod
    def apply_loss(loss):
        # Loss channel
        loss.build_training_network()
        # Evaluation channel
        loss.build_evaluation_network()
        # Finalize, clean up
        loss.finalize()
