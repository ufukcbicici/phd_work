import tensorflow as tf
import numpy as np

from collections import deque

from algorithms.custom_batch_norm_algorithms import CustomBatchNormAlgorithms
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.cign.fast_tree_multi_gpu import FastTreeMultiGpu
from simple_tf.global_params import GlobalConstants
from simple_tf.info_gain import InfoGainLoss
from simple_tf.node import Node


class CignMultiGpu(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset):
        super().__init__(node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset)
        # Each element contains a (device_str, network) pair.
        self.towerNetworks = []
        devices = UtilityFuncs.get_available_devices(only_gpu=False)
        device_count = len(devices)
        assert GlobalConstants.BATCH_SIZE % device_count == 0
        tower_batch_size = GlobalConstants.BATCH_SIZE / len(devices)
        for tower_id, device_str in enumerate(devices):
            with tf.device(device_str):
                tower_cign = FastTreeMultiGpu(
                    node_build_funcs=self.nodeBuildFuncs,
                    grad_func=self.gradFunc,
                    threshold_func=self.thresholdFunc,
                    residue_func=self.residueFunc,
                    summary_func=self.summaryFunc,
                    degree_list=self.degreeList,
                    dataset=dataset,
                    container_network=self,
                    tower_id=tower_id,
                    tower_batch_size=tower_batch_size)
                tower_cign.build_network()
                self.towerNetworks.append((device_str, tower_cign))
            tf.get_variable_scope().reuse_variables()
        all_vars = tf.global_variables()
        # Assert that all variables are created on the CPU memory.
        assert all(["CPU" in var.device and "GPU" not in var.device for var in all_vars])

    def build_towers(self):
        for device_str, tower_cign in self.towerNetworks:
            print("X")
