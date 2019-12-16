import numpy as np
import tensorflow as tf

from algorithms.custom_batch_norm_algorithms import CustomBatchNormAlgorithms
from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.cign_multi_gpu import CignMultiGpu
from simple_tf.cign.cign_single_late_exit import CignSingleLateExit
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants, AccuracyCalcType


class CignMultiGpuSingleLateExit(CignMultiGpu):
    def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                 dataset, network_name, late_exit_func):
        super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                         dataset, network_name)
        self.lateExitFunc = late_exit_func

    def get_tower_network(self):
        tower_cign = CignSingleLateExit(
            node_build_funcs=self.nodeBuildFuncs,
            grad_func=None,
            hyperparameter_func=None,
            residue_func=None,
            summary_func=None,
            degree_list=self.degreeList,
            dataset=self.dataset,
            network_name=self.networkName,
            late_exit_func=self.lateExitFunc)
        return tower_cign
