import numpy as np
import tensorflow as tf

from algorithms.custom_batch_norm_algorithms import CustomBatchNormAlgorithms
from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants, AccuracyCalcType


class CignMultiGpu(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                 dataset, network_name):
        GlobalConstants.USE_MULTI_GPU = True
        # placeholders = [op for op in tf.get_default_graph().get_operations() if op.type == "Placeholder"]
        with tf.name_scope("main_network"):
            super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                             dataset, network_name)
        # Each element contains a (device_str, network) pair.
        self.towerNetworks = []
        self.grads = []
        self.averagedGrads = None
        self.dataset = dataset
        self.applyGradientsOp = None
        self.batchNormMovingAvgAssignOps = []
        # Unit test variables
        self.batchNormMovingAverageValues = {}

    # def build_towers(self):
    #     for device_str, tower_cign in self.towerNetworks:
    #         print("X")

    def build_optimizer(self):
        # Build optimizer
        # self.globalCounter = tf.Variable(0, trainable=False)
        with tf.device("/device:CPU:0"):
            self.globalCounter = tf.get_variable("global_counter", initializer=0, dtype=tf.int32, trainable=False)
            # self.globalCounter =
            # UtilityFuncs.create_variable(name="global_counter", shape=[], initializer=0,
            # trainable=False, dtype=tf.int32)
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

    def get_tower_network(self):
        tower_cign = FastTreeNetwork(
            node_build_funcs=self.nodeBuildFuncs,
            grad_func=None,
            hyperparameter_func=None,
            residue_func=None,
            summary_func=None,
            degree_list=self.degreeList,
            dataset=self.dataset,
            network_name=self.networkName)
        return tower_cign

    def build_network(self):
        devices = UtilityFuncs.get_available_devices(only_gpu=(not GlobalConstants.USE_CPU_AS_DEVICE))
        print(devices)
        device_count = len(devices)
        assert GlobalConstants.BATCH_SIZE % device_count == 0
        with tf.device(GlobalConstants.GLOBAL_PINNING_DEVICE):
            with tf.variable_scope("multiple_networks"):
                self.build_optimizer()
                for tower_id, device_str in enumerate(devices):
                    with tf.device(device_str):
                        with tf.name_scope("tower_{0}".format(tower_id)):
                            print(device_str)
                            # Build a Multi GPU supporting CIGN
                            tower_cign = self.get_tower_network()
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
                self.averagedGrads = self.average_gradients()
                # Apply the gradients to adjust the shared variables.
                with tf.control_dependencies(self.batchNormMovingAvgAssignOps):
                    self.applyGradientsOp = self.optimizer.apply_gradients(self.averagedGrads,
                                                                           global_step=self.globalCounter)
        # Unify all evaluation tensors across towers
        self.prepare_evaluation_dictionary()
        # placeholders = [op for op in tf.get_default_graph().get_operations() if op.type == "Placeholder"]
        # all_vars = tf.global_variables()
        # Assert that all variables are created on the CPU memory.
        # assert all(["CPU" in var.device for var in all_vars])
        self.dataset = None
        self.topologicalSortedNodes = self.towerNetworks[0][1].topologicalSortedNodes

    def prepare_evaluation_dictionary(self):
        self.evalDict = {}
        for tower_id, tpl in enumerate(self.towerNetworks):
            network = tpl[1]
            for k, v in network.evalDict.items():
                new_key = "tower_{0}_{1}".format(tower_id, k)
                self.evalDict[new_key] = v
        batch_norm_moving_averages = tf.get_collection(CustomBatchNormAlgorithms.CUSTOM_BATCH_NORM_OPS)
        for tpl in batch_norm_moving_averages:
            if tpl[0].name not in self.evalDict:
                self.evalDict[tpl[0].name] = []
            self.evalDict[tpl[0].name].append(tpl)
        self.sampleCountTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "sample_count" in k}
        self.isOpenTensors = {k: self.evalDict[k] for k in self.evalDict.keys() if "is_open" in k}
        self.infoGainDicts = {k: v for k, v in self.evalDict.items() if "info_gain" in k}

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
        batch_norm_moving_averages = tf.get_collection(CustomBatchNormAlgorithms.CUSTOM_BATCH_NORM_OPS)
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

        # # Average updates for tensorflow batch normalization operations
        # tensorflow_batch_norm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # moving_avg_variables = [v for v in tf.global_variables() if "moving_" in v.name]
        # self.batchNormMovingAvgAssignOps = []
        # for moving_avg_var in moving_avg_variables:
        #     var_name = moving_avg_var.name
        #     matched_update_ops = [op.inputs for op in tensorflow_batch_norm_ops if op.inputs[0].name == var_name]
        #     assert len(matched_update_ops) == len(self.towerNetworks)
        #     assert all([matched_update_ops[i][0] == matched_update_ops[0][0] for i in range(len(matched_update_ops))])
        #     updates = []
        #     for tpl in matched_update_ops:
        #         delta = tpl[1]
        #         expanded_delta = tf.expand_dims(delta, 0)
        #         updates.append(expanded_delta)
        #     update = tf.concat(axis=0, values=updates)
        #     update = tf.reduce_mean(update, 0)
        #     assign_sub_op = tf.assign_sub(moving_avg_var, update)
        #     self.batchNormMovingAvgAssignOps.append(assign_sub_op)
        # # Average updates for custom batch normalization operations
        # custom_batch_norm_ops = tf.get_collection(CustomBatchNormAlgorithms.CUSTOM_BATCH_NORM_OPS)
        # batch_norm_ops_dict = {}
        # for moving_average, new_value in custom_batch_norm_ops:
        #     if moving_average not in batch_norm_ops_dict:
        #         batch_norm_ops_dict[moving_average] = []
        #     expanded_new_value = tf.expand_dims(new_value, 0)
        #     batch_norm_ops_dict[moving_average].append(expanded_new_value)
        # assert all([len(v) == len(self.towerNetworks) for k, v in batch_norm_ops_dict.items()])
        # # Take the mean of all values for every moving average and update the moving average value.
        # for moving_average, values_list in batch_norm_ops_dict.items():
        #     values_concat = tf.concat(axis=0, values=values_list)
        #     mean_new_value = tf.reduce_mean(values_concat, 0)
        #     momentum = GlobalConstants.BATCH_NORM_DECAY
        #     new_moving_average_value = tf.where(self.iterationHolder > 0,
        #                                         (momentum * moving_average + (1.0 - momentum) * mean_new_value),
        #                                         mean_new_value)
        #     new_moving_average_value_assign_op = tf.assign(moving_average, new_moving_average_value)
        #     self.batchNormMovingAvgAssignOps.append(new_moving_average_value_assign_op)
        # print("X")

    # OK
    def get_probability_thresholds(self, feed_dict, iteration, update):
        for tower_id, tpl in enumerate(self.towerNetworks):
            network = tpl[1]
            for node in network.topologicalSortedNodes:
                if node.isLeaf:
                    continue
                if update:
                    # Probability Threshold
                    node_degree = network.degreeList[node.depth]
                    uniform_prob = 1.0 / float(node_degree)
                    threshold = uniform_prob - node.probThresholdCalculator.value
                    feed_dict[node.probabilityThreshold] = threshold
                    print("{0} value={1}".format(node.probThresholdCalculator.name, threshold))
                    # Update the threshold calculator
                    node.probThresholdCalculator.update(iteration=iteration + 1)
                else:
                    feed_dict[node.probabilityThreshold] = 0.0

    # OK
    def get_softmax_decays(self, feed_dict, iteration, update):
        for tower_id, tpl in enumerate(self.towerNetworks):
            network = tpl[1]
            for node in network.topologicalSortedNodes:
                if node.isLeaf:
                    continue
                # Decay for Softmax
                decay = node.softmaxDecayCalculator.value
                if update:
                    feed_dict[node.softmaxDecay] = decay
                    print("{0} value={1}".format(node.softmaxDecayCalculator.name, decay))
                    # Update the Softmax Decay
                    node.softmaxDecayCalculator.update(iteration=iteration + 1)
                else:
                    feed_dict[node.softmaxDecay] = GlobalConstants.SOFTMAX_TEST_TEMPERATURE

    # OK
    def get_decision_dropout_prob(self, feed_dict, iteration, update):
        for tower_id, tpl in enumerate(self.towerNetworks):
            network = tpl[1]
            if update:
                prob = network.decisionDropoutKeepProbCalculator.value
                feed_dict[network.decisionDropoutKeepProb] = prob
                print("{0} value={1}".format(network.decisionDropoutKeepProbCalculator.name, prob))
                network.decisionDropoutKeepProbCalculator.update(iteration=iteration + 1)
            else:
                feed_dict[network.decisionDropoutKeepProb] = 1.0

    # OK
    def get_decision_weight(self, feed_dict, iteration, update):
        for tower_id, tpl in enumerate(self.towerNetworks):
            network = tpl[1]
            weight = network.decisionLossCoefficientCalculator.value
            feed_dict[network.decisionLossCoefficient] = weight
            # print("self.decisionLossCoefficient={0}".format(weight))
            if update:
                network.decisionLossCoefficientCalculator.update(iteration=iteration + 1)

    def prepare_feed_dict(self, minibatch, iteration, use_threshold, is_train, use_masking):
        # Load the placeholders in each tower separately
        actual_batch_size = minibatch.samples.shape[0]
        assert actual_batch_size % len(self.towerNetworks) == 0
        single_tower_batch_size = actual_batch_size / len(self.towerNetworks)
        feed_dict = {self.iterationHolder: iteration}
        # Global parameters
        for tower_id, tpl in enumerate(self.towerNetworks):
            device_str = tpl[0]
            network = tpl[1]
            lower_bound = int(tower_id * single_tower_batch_size)
            upper_bound = int((tower_id + 1) * single_tower_batch_size)
            feed_dict[network.dataTensor] = minibatch.samples[lower_bound:upper_bound]
            feed_dict[network.labelTensor] = minibatch.labels[lower_bound:upper_bound]
            feed_dict[network.indicesTensor] = minibatch.indices[lower_bound:upper_bound]
            feed_dict[network.oneHotLabelTensor] = minibatch.one_hot_labels[lower_bound:upper_bound]
            feed_dict[network.weightDecayCoeff] = GlobalConstants.WEIGHT_DECAY_COEFFICIENT
            feed_dict[network.decisionWeightDecayCoeff] = GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT
            feed_dict[network.useThresholding] = int(use_threshold)
            feed_dict[network.isTrain] = int(is_train)
            feed_dict[network.useMasking] = int(use_masking)
            feed_dict[network.informationGainBalancingCoefficient] = GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT
            feed_dict[network.iterationHolder] = iteration
            feed_dict[network.filteredMask] = np.ones((int(single_tower_batch_size),), dtype=bool)
            if is_train:
                feed_dict[network.classificationDropoutKeepProb] = GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB
            else:
                feed_dict[network.classificationDropoutKeepProb] = 1.0
        # Per network parameters
        if not self.isBaseline:
            if is_train:
                self.get_probability_thresholds(feed_dict=feed_dict, iteration=iteration, update=True)
                self.get_softmax_decays(feed_dict=feed_dict, iteration=iteration, update=True)
                self.get_decision_dropout_prob(feed_dict=feed_dict, iteration=iteration, update=True)
                self.get_decision_weight(feed_dict=feed_dict, iteration=iteration, update=True)
            else:
                self.get_probability_thresholds(feed_dict=feed_dict, iteration=1000000, update=False)
                self.get_softmax_decays(feed_dict=feed_dict, iteration=1000000, update=False)
                self.get_decision_dropout_prob(feed_dict=feed_dict, iteration=1000000, update=False)
                self.get_decision_weight(feed_dict=feed_dict, iteration=1000000, update=False)
        return feed_dict

    def get_run_ops(self):
        network_losses = [tpl[1].finalLoss for tpl in self.towerNetworks]
        run_ops = [self.applyGradientsOp, self.learningRate, self.sampleCountTensors, self.isOpenTensors,
                   self.infoGainDicts, self.grads, self.averagedGrads, network_losses]

        # custom_batch_norm_ops = {}
        # for k, v in tf.get_collection(CustomBatchNormAlgorithms.CUSTOM_BATCH_NORM_OPS):
        #     if k.name not in custom_batch_norm_ops:
        #         custom_batch_norm_ops[k.name] = []
        #     custom_batch_norm_ops[k.name].append(v)
        # run_ops.append(custom_batch_norm_ops)

        # custom_batch_norm_ops = [(tpl[0].name, tpl[1])
        #                          for tpl in tf.get_collection(CustomBatchNormAlgorithms.CUSTOM_BATCH_NORM_OPS)]
        # run_ops.append(custom_batch_norm_ops)
        # run_ops.extend(self.batchNormMovingAvgAssignOps)
        # run_ops = [self.learningRate, self.sampleCountTensors, self.isOpenTensors,
        #            self.infoGainDicts]
        return run_ops

    def unit_test_batch_norm_ops(self, sess, eval_results):
        batch_norm_moving_averages = tf.get_collection(CustomBatchNormAlgorithms.CUSTOM_BATCH_NORM_OPS)
        moving_average_curr_value_tensors = {}
        for tpl in batch_norm_moving_averages:
            moving_average_variable = tpl[0]
            if moving_average_variable.name not in moving_average_curr_value_tensors:
                moving_average_curr_value_tensors[moving_average_variable.name] = moving_average_variable
            else:
                assert moving_average_variable == moving_average_curr_value_tensors[moving_average_variable.name]
        momentum = GlobalConstants.BATCH_NORM_DECAY
        res = sess.run(moving_average_curr_value_tensors)
        for var_name in moving_average_curr_value_tensors.keys():
            values = eval_results[var_name]
            # values = eval_results[var_name]
            tf_moving_average_value = res[var_name]
            new_value_arrays = [np.expand_dims(v, axis=0) for v in values]
            unified_arr = np.concatenate(new_value_arrays, axis=0)
            mean_arr = np.mean(unified_arr, axis=0)
            if var_name not in self.batchNormMovingAverageValues:
                self.batchNormMovingAverageValues[var_name] = mean_arr
            else:
                curr_value = self.batchNormMovingAverageValues[var_name]
                self.batchNormMovingAverageValues[var_name] = momentum * curr_value + (1.0 - momentum) * mean_arr
            assert np.allclose(self.batchNormMovingAverageValues[var_name], tf_moving_average_value)
            # if not np.allclose(self.batchNormMovingAverageValues[var_name], tf_moving_average_value):
            #     print("X")

    def get_explanation_string(self):
        explanation = ""
        explanation += "TOTAL_EPOCH_COUNT:{0}\n".format(GlobalConstants.TOTAL_EPOCH_COUNT)
        explanation += "EPOCH_COUNT:{0}\n".format(GlobalConstants.EPOCH_COUNT)
        explanation += "EPOCH_REPORT_PERIOD:{0}\n".format(GlobalConstants.EPOCH_REPORT_PERIOD)
        explanation += "BATCH_SIZE:{0}\n".format(GlobalConstants.BATCH_SIZE)
        explanation += "EVAL_BATCH_SIZE:{0}\n".format(GlobalConstants.EVAL_BATCH_SIZE)
        explanation += "USE_MULTI_GPU:{0}\n".format(GlobalConstants.USE_MULTI_GPU)
        explanation += "USE_SAMPLING_CIGN:{0}\n".format(GlobalConstants.USE_SAMPLING_CIGN)
        explanation += "USE_RANDOM_SAMPLING:{0}\n".format(GlobalConstants.USE_RANDOM_SAMPLING)
        explanation += "LR SCHEDULE:{0}\n".format(GlobalConstants.LEARNING_RATE_CALCULATOR.get_explanation())
        explanation += "USE_SCALED_GRADIENTS:{0}\n".format(GlobalConstants.USE_SCALED_GRADIENTS)
        network = self.towerNetworks[0][1]
        leaf_node = [node for node in network.topologicalSortedNodes if node.isLeaf][0]
        root_to_leaf_path = network.dagObject.ancestors(node=leaf_node)
        root_to_leaf_path.append(leaf_node)
        path_mac_cost = sum([node.macCost for node in root_to_leaf_path])
        explanation += "Mac Cost:{0}\n".format(path_mac_cost)
        explanation += "Mac Cost per Nodes:{0}\n".format(network.nodeCosts)
        return explanation

    def calculate_model_performance(self, sess, dataset, run_id, epoch_id, iteration):
        network = self.towerNetworks[0][1]
        network.calculate_model_performance(sess=sess, dataset=dataset, run_id=run_id, epoch_id=epoch_id,
                                            iteration=iteration)
        # # moving_results_1 = sess.run(moving_stat_vars)
        # is_evaluation_epoch_at_report_period = \
        #     epoch_id < GlobalConstants.TOTAL_EPOCH_COUNT - GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING \
        #     and (epoch_id + 1) % GlobalConstants.EPOCH_REPORT_PERIOD == 0
        # is_evaluation_epoch_before_ending = \
        #     epoch_id >= GlobalConstants.TOTAL_EPOCH_COUNT - GlobalConstants.EVALUATION_EPOCHS_BEFORE_ENDING
        # if is_evaluation_epoch_at_report_period or is_evaluation_epoch_before_ending:
        #     training_accuracy, training_confusion = \
        #         network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.training,
        #                                    run_id=run_id,
        #                                    iteration=iteration)
        #     validation_accuracy, validation_confusion = \
        #         network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.test,
        #                                    run_id=run_id,
        #                                    iteration=iteration)
        #     validation_accuracy_corrected, validation_marginal_corrected = \
        #         network.accuracyCalculator.calculate_accuracy_with_route_correction(
        #             sess=sess, dataset=dataset,
        #             dataset_type=DatasetTypes.test)
        #     if is_evaluation_epoch_before_ending:
        #         self.save_model(sess=sess, run_id=run_id, iteration=iteration)
        #         network.save_routing_info(sess=sess, run_id=run_id, iteration=iteration,
        #                                   dataset=dataset, dataset_type=DatasetTypes.test)
        #         # network.test_save_load(sess=sess, run_id=run_id, iteration=iteration,
        #         #                        dataset=dataset, dataset_type=DatasetTypes.test)
        #     DbLogger.write_into_table(
        #         rows=[(run_id, iteration, epoch_id, training_accuracy,
        #                validation_accuracy, validation_accuracy_corrected,
        #                0.0, 0.0, "XXX")], table=DbLogger.logsTable, col_count=9)
