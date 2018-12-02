from collections import deque

import tensorflow as tf
import numpy as np
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.node import Node


class Jungle(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset):
        super().__init__(node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset)
        curr_index = 0
        is_leaf = 0 == (self.depth - 1)
        root_node = Node(index=curr_index, depth=0, is_root=True, is_leaf=is_leaf)
        threshold_name = self.get_variable_name(name="threshold", node=root_node)
        root_node.probabilityThreshold = tf.placeholder(name=threshold_name, dtype=tf.float32)
        softmax_decay_name = self.get_variable_name(name="softmax_decay", node=root_node)
        root_node.softmaxDecay = tf.placeholder(name=softmax_decay_name, dtype=tf.float32)
        self.dagObject.add_node(node=root_node)
        self.nodes[curr_index] = root_node
        d = deque()
        d.append(root_node)
