import tensorflow as tf
from auxillary.constants import ArgumentTypes
from framework.network_argument import NetworkArgument


class NetworkNode:
    def __init__(self, index, containing_network, is_root, is_leaf):
        self.index = index
        self.indicatorText = "Node_{0}".format(self.index)
        self.argumentsDict = {}
        self.parentNetwork = containing_network
        self.isRoot = is_root
        self.isLeaf = is_leaf
        self.networkChannels = {}

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
