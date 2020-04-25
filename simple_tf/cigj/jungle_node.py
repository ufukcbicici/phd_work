import tensorflow as tf
from enum import Enum

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.uncategorized.node import Node


class NodeType(Enum):
    root_node = 0
    f_node = 1
    h_node = 2
    leaf_node = 3


class JungleNode(Node):
    def __init__(self, index, depth, node_type):
        is_root = node_type == NodeType.root_node
        is_leaf = node_type == NodeType.leaf_node
        self.nodeType = node_type
        self.F_input = None
        self.H_input = None
        self.F_output = None
        self.H_output = None
        self.partitioned_F_outputs = None
        self.partitioned_H_outputs = None
        self.conditionIndices = None
        self.conditionProbabilities = None
        self.sampleCountTensor = None
        self.gumbelSoftmaxTemperatureCalculator = None
        # This is for debugging purposes
        self.stitchedIndices = None
        self.stitchedLabels = None
        super().__init__(index, depth, is_root, is_leaf)
        self.softmaxDecay = tf.placeholder(name=UtilityFuncs.get_variable_name(name="softmax_decay", node=self),
                                           dtype=tf.float32)
        self.gumbelSoftmaxTemperature = tf.placeholder(
            name=UtilityFuncs.get_variable_name(name="gumbelSoftmaxTemperature", node=self),
            dtype=tf.float32)
        if self.nodeType == NodeType.h_node:
            self.partitioned_F_outputs = []
            self.partitioned_H_outputs = []
            self.conditionIndices = []
