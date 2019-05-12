import tensorflow as tf
import numpy as np

from collections import deque
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.node import Node


class CignMultiGpu(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset):
        super().__init__(node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset)

    def build_network(self):
        # Build the tree topologically and create the Tensorflow placeholders
        self.build_tree()
