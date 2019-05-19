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
        self.grads = []
        self.dataset = dataset

    # def build_towers(self):
    #     for device_str, tower_cign in self.towerNetworks:
    #         print("X")

    def build_optimizer(self):
        # Build optimizer
        # self.globalCounter = tf.Variable(0, trainable=False)
        self.globalCounter = UtilityFuncs.create_variable(name="global_counter",
                                                          shape=[], initializer=0, trainable=False, dtype=tf.int32)
        boundaries = [tpl[0] for tpl in GlobalConstants.LEARNING_RATE_CALCULATOR.schedule]
        values = [GlobalConstants.INITIAL_LR]
        values.extend([tpl[1] for tpl in GlobalConstants.LEARNING_RATE_CALCULATOR.schedule])
        self.learningRate = tf.train.piecewise_constant(self.globalCounter, boundaries, values)
        self.optimizer = tf.train.MomentumOptimizer(self.learningRate, 0.9)
        # self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # # pop_var = tf.Variable(name="pop_var", initial_value=tf.constant(0.0, shape=(16, )), trainable=False)
        # # pop_var_assign_op = tf.assign(pop_var, tf.constant(45.0, shape=(16, )))
        # with tf.control_dependencies(self.extra_update_ops):
        #     self.optimizer = tf.train.MomentumOptimizer(self.learningRate, 0.9).minimize(self.finalLoss,
        #                                                                                  global_step=self.globalCounter)

    def build_network(self):
        devices = UtilityFuncs.get_available_devices(only_gpu=False)
        device_count = len(devices)
        assert GlobalConstants.BATCH_SIZE % device_count == 0
        tower_batch_size = GlobalConstants.BATCH_SIZE / len(devices)
        self.build_optimizer()
        for tower_id, device_str in enumerate(devices):
            with tf.device(device_str):
                # Build a Multi GPU supporting CIGN
                tower_cign = FastTreeMultiGpu(
                    node_build_funcs=self.nodeBuildFuncs,
                    grad_func=self.gradFunc,
                    threshold_func=self.thresholdFunc,
                    residue_func=self.residueFunc,
                    summary_func=self.summaryFunc,
                    degree_list=self.degreeList,
                    dataset=self.dataset,
                    container_network=self,
                    tower_id=tower_id,
                    tower_batch_size=tower_batch_size)
                tower_cign.build_network()
                self.towerNetworks.append((device_str, tower_cign))
                # Calculate gradients
                tower_grads = self.optimizer.compute_gradients(loss=tower_cign.finalLoss)
                # Assert that all gradients are correctly calculated.
                assert all([tpl[0] is not None for tpl in tower_grads])
                assert all([tpl[1] is not None for tpl in tower_grads])
                self.grads.append(tower_grads)
            tf.get_variable_scope().reuse_variables()
        all_vars = tf.global_variables()
        # Assert that all variables are created on the CPU memory.
        assert all(["CPU" in var.device and "GPU" not in var.device for var in all_vars])
        self.dataset = None