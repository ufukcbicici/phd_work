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
        self.applyGradientsOp = None
        self.batchNormMovingAvgAssignOps = []

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
        with tf.device('/CPU:0'):
            self.build_optimizer()
            with tf.variable_scope("multiple_networks"):
                for tower_id, device_str in enumerate(devices):
                    with tf.device(device_str):
                        with tf.name_scope("tower_{0}".format(tower_id)):
                            print(device_str)
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
                            print("Built network for tower {0}".format(tower_id))
                            self.towerNetworks.append((device_str, tower_cign))
                            # Calculate gradients
                            tower_grads = self.optimizer.compute_gradients(loss=tower_cign.finalLoss)
                            # Assert that all gradients are correctly calculated.
                            assert all([tpl[0] is not None for tpl in tower_grads])
                            assert all([tpl[1] is not None for tpl in tower_grads])
                            self.grads.append(tower_grads)
                    var_scope = tf.get_variable_scope()
                    var_scope.reuse_variables()
            with tf.variable_scope("optimizer"):
                # Calculate the mean of the moving average updates for batch normalization operations, across each tower.
                self.prepare_batch_norm_moving_avg_ops()
                # We must calculate the mean of each gradient.
                # Note that this is the synchronization point across all towers.
                grads = self.average_gradients()
                # Apply the gradients to adjust the shared variables.
                self.applyGradientsOp = self.optimizer.apply_gradients(grads, global_step=self.globalCounter)
        all_vars = tf.global_variables()
        # Assert that all variables are created on the CPU memory.
        assert all(["CPU" in var.device and "GPU" not in var.device for var in all_vars])
        self.dataset = None

    def average_gradients(self):
        average_grads = []
        for grad_and_vars in zip(*self.grads):
            # Each grad_and_vars is of the form: ((grad0_gpu0, var0_gpu0), ..., (grad0_gpuN, var0_gpuN))
            grads = []
            _vars = []
            for g, v in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
                _vars.append(v)
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            # Assert that all variables are the same, verify variable sharing behavior over towers.
            _var = _vars[0]
            assert all([v == _var for v in _vars])
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            grad_and_var = (grad, _var)
            average_grads.append(grad_and_var)
        return average_grads

    def prepare_batch_norm_moving_avg_ops(self):
        batch_norm_moving_averages = tf.get_collection(CustomBatchNormAlgorithms.BATCH_NORM_OPS)
        # Assert that for every (moving_average, new_value) tuple, we have exactly #tower_count tuples with a specific
        # moving_average entry.
        batch_norm_ops_dict = {}
        for moving_average, new_value in batch_norm_moving_averages:
            if moving_average not in batch_norm_ops_dict:
                batch_norm_ops_dict[moving_average] = []
            expanded_new_value = tf.expand_dims(new_value, 0)
            batch_norm_ops_dict[moving_average].append(expanded_new_value)
        assert all([len(v) == len(self.towerNetworks) for k, v in batch_norm_ops_dict.items()])
        # Take the mean of all values for every moving average and update the moving average value.
        for moving_average, values_list in batch_norm_ops_dict.items():
            values_concat = tf.concat(axis=0, values=values_list)
            mean_new_value = tf.reduce_mean(values_concat, 0)
            momentum = GlobalConstants.BATCH_NORM_DECAY
            new_moving_average_value = tf.where(self.iterationHolder > 0,
                                                (momentum * moving_average + (1.0 - momentum) * mean_new_value),
                                                mean_new_value)
            new_moving_average_value_assign_op = tf.assign(moving_average, new_moving_average_value)
            self.batchNormMovingAvgAssignOps.append(new_moving_average_value_assign_op)
