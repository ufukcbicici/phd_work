import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from algorithms.accuracy_calculator import AccuracyCalculator
from auxillary.dag_utilities import Dag
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cigj.jungle import Jungle
from simple_tf.cigj.jungle_node import JungleNode
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants
from simple_tf.info_gain import InfoGainLoss


class JungleNoStitch(Jungle):
    def __init__(self, node_build_funcs, h_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list,
                 dataset):
        super().__init__(node_build_funcs, h_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list,
                         dataset)


