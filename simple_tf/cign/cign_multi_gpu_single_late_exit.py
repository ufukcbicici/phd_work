import tensorflow as tf
from algorithms.custom_batch_norm_algorithms import CustomBatchNormAlgorithms
from simple_tf.cign.cign_multi_gpu import CignMultiGpu
from simple_tf.cign.cign_single_late_exit import CignSingleLateExit
from simple_tf.global_params import GlobalConstants


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

    def prepare_feed_dict(self, minibatch, iteration, use_threshold, is_train, use_masking):
        feed_dict = super().prepare_feed_dict(minibatch=minibatch, iteration=iteration, use_threshold=use_threshold,
                                              is_train=is_train, use_masking=use_masking)
        for tower_id, tpl in enumerate(self.towerNetworks):
            device_str = tpl[0]
            network = tpl[1]
            feed_dict[network.earlyExitWeight] = GlobalConstants.EARLY_EXIT_WEIGHT
            feed_dict[network.lateExitWeight] = GlobalConstants.LATE_EXIT_WEIGHT
        return feed_dict

    def prepare_batch_norm_moving_avg_ops(self):
        batch_norm_moving_averages = tf.get_collection(CustomBatchNormAlgorithms.CUSTOM_BATCH_NORM_OPS)
        # Assert that for every (moving_average, new_value) tuple, we have exactly #tower_count tuples with a specific
        # moving_average entry.
        batch_norm_ops_dict = {}
        for moving_average, new_value in batch_norm_moving_averages:
            if "late_exit_test" in new_value.name:
                continue
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