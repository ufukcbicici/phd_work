import tensorflow as tf
import numpy as np


class NetworkArgument:
    def __init__(self, name, symbolic_network_object, container_node, needs_gradient, arg_type):
        self.name = name
        self.tensor = symbolic_network_object
        self.containerNode = container_node
        self.needsGradient = needs_gradient
        self.valueArray = None
        self.gradientArray = None
        self.gradientIndex = None
        self.argumentType = arg_type

    def set_value(self, arr_ref):
        self.valueArray = arr_ref

    def set_gradient(self, arr_ref):
        self.gradientArray = arr_ref
