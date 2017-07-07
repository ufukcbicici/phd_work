from auxillary.constants import ArgumentTypes
from framework.network_argument import NetworkArgument


class NetworkNode:
    def __init__(self, index):
        self.index = index
        self.indicatorText = "Node_{0}".format(self.index)
        self.argumentsDict = {}

    def create_variable(self, name, tf_object, needs_gradient, arg_type):
        argument_name = "{0}_{1}".format(self.indicatorText, name)
        if argument_name in self.argumentsDict:
            raise Exception("Another argument with name {0} exists.".format(argument_name))
        argument = NetworkArgument(name=argument_name, tf_object=tf_object, container_node=self,
                                   needs_gradient=needs_gradient,  arg_type=arg_type)
        self.argumentsDict[argument_name] = argument
