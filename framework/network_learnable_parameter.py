import tensorflow as tf
import numpy as np

from auxillary.constants import GlobalInputNames, ChannelTypes, LossType
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

    def get_gradient(self, loss_type=LossType.objective):
        if loss_type == LossType.objective:
            grad = \
                self.containerNode.parentNetwork.trainingResults[ChannelTypes.objective_gradients.value][
                    self.globalParameterIndex]
        elif loss_type == LossType.regularization:
            grad = \
                self.containerNode.parentNetwork.trainingResults[ChannelTypes.regularization_gradients.value][
                    self.globalParameterIndex]
        else:
            raise NotImplementedError()
        return grad

    def get_property_name(self, property_):
        return "{0}_{1}".format(property_, self.name)
