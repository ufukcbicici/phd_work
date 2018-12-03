import tensorflow as tf
from enum import Enum

from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.node import Node


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
        super().__init__(index, depth, is_root, is_leaf)
