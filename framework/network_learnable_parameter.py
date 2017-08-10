import tensorflow as tf
import numpy as np

from auxillary.constants import GlobalInputNames, ChannelTypes
from framework.network_channel import NetworkChannel


class NetworkLearnableParameter:
    def __init__(self, name, symbolic_network_object, container_node, arg_type):
        self.name = name
        self.inputName = "{0}_{1}".format(self.name, GlobalInputNames.parameter_update.value)
        self.tensor = symbolic_network_object
        self.containerNode = container_node
        self.valueArray = None
        self.gradientArray = None
        self.globalParameterIndex = None
        self.parameterType = arg_type
        self.assignOp = None
        # Create update mechanism for the parameter tensor
        self.inputTensor = \
            self.containerNode.parentNetwork.add_networkwise_input(name=self.inputName, tensor_type=self.tensor.dtype)
        with NetworkChannel(parent_node=self.containerNode,
                            parent_node_channel=ChannelTypes.parameter_update) as param_update_channel:
            self.assignOp = param_update_channel.add_operation(op=tf.assign(self.tensor, self.inputTensor))

    def set_value(self, arr_ref):
        self.valueArray = arr_ref

    def set_gradient(self, arr_ref):
        self.gradientArray = arr_ref

    def get_property_name(self, property_):
        return "{0}_{1}".format(property_, self.name)
