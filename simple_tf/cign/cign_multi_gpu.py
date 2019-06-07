import tensorflow as tf
import numpy as np

from collections import deque

from algorithms.custom_batch_norm_algorithms import CustomBatchNormAlgorithms
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.cign.fast_tree_multi_gpu import FastTreeMultiGpu
from simple_tf.global_params import GlobalConstants, AccuracyCalcType
from auxillary.parameters import FixedParameter, DiscreteParameter, DecayingParameter
from simple_tf.info_gain import InfoGainLoss
from simple_tf.node import Node


class CignMultiGpu(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                 dataset):
        # placeholders = [op for op in tf.get_default_graph().get_operations() if op.type == "Placeholder"]
        with tf.name_scope("main_network"):
            super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
                             dataset)
        # Each element contains a (device_str, network) pair.
        self.towerNetworks = []
        self.grads = []
        self.dataset = dataset
        self.applyGradientsOp = None
        self.batchNormMovingAvgAssignOps = []
        self.towerBatchSize = None
        # Unit test variables
        self.batchNormMovingAverageValues = {}

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
        devices = UtilityFuncs.get_available_devices(only_gpu=True)
        device_count = len(devices)
        assert GlobalConstants.BATCH_SIZE % device_count == 0
        self.towerBatchSize = GlobalConstants.BATCH_SIZE / len(devices)
        with tf.device('/CPU:0'):
            with tf.variable_scope("multiple_networks"):
                self.build_optimizer()
                for tower_id, device_str in enumerate(devices):
                    with tf.device(device_str):
                        with tf.name_scope("tower_{0}".format(tower_id)):
                            print(device_str)
                            # Build a Multi GPU supporting CIGN
                            tower_cign = FastTreeMultiGpu(
                                node_build_funcs=self.nodeBuildFuncs,
                                grad_func=self.gradFunc,
                                hyperparameter_func=self.hyperparameterFunc,
                                residue_func=self.residueFunc,
                                summary_func=self.summaryFunc,
                                degree_list=self.degreeList,
                                dataset=self.dataset,
                                container_network=self,
                                tower_id=tower_id,
                                tower_batch_size=self.towerBatchSize)
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
                with tf.control_dependencies(self.batchNormMovingAvgAssignOps):
                    self.applyGradientsOp = self.optimizer.apply_gradients(grads, global_step=self.globalCounter)
        # Unify all evaluation tensors across towers
        self.prepare_evaluation_dictionary()
        placeholders = [op for op in tf.get_default_graph().get_operations() if op.type == "Placeholder"]
        all_vars = tf.global_variables()
        # Assert that all variables are created on the CPU memory.
        assert all(["CPU" in var.device and "GPU" not in var.device for var in all_vars])
        self.dataset = None
        self.topologicalSortedNodes = self.towerNetworks[0][1].topologicalSortedNodes

    def prepare_evaluation_dictionary(self):
        self.evalDict = {}
        for tower_id, tpl in enumerate(self.towerNetworks):
            network = tpl[1]
            for k, v in network.evalDict.items():
                new_key = "tower_{0}_{1}".format(tower_id, k)
                self.evalDict[new_key] = v
        batch_norm_moving_averages = tf.get_collection(CustomBatchNormAlgorithms.BATCH_NORM_OPS)
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

    def set_hyperparameters(self, **kwargs):
        GlobalConstants.WEIGHT_DECAY_COEFFICIENT = kwargs["weight_decay_coefficient"]
        GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB = kwargs["classification_keep_probability"]
        if not self.isBaseline:
            GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT = kwargs["decision_weight_decay_coefficient"]
            GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT = kwargs["info_gain_balance_coefficient"]
            for tower_id, tpl in enumerate(self.towerNetworks):
                network = tpl[1]
                network.decisionDropoutKeepProbCalculator = FixedParameter(name="decision_dropout_prob",
                                                                           value=kwargs["decision_keep_probability"])

                # Noise Coefficient
                network.noiseCoefficientCalculator = DecayingParameter(name="noise_coefficient_calculator", value=0.0,
                                                                       decay=0.0,
                                                                       decay_period=1,
                                                                       min_limit=0.0)
                # Decision Loss Coefficient
                # network.decisionLossCoefficientCalculator = DiscreteParameter(name="decision_loss_coefficient_calculator",
                #                                                               value=0.0,
                #                                                               schedule=[(12000, 1.0)])
                network.decisionLossCoefficientCalculator = FixedParameter(name="decision_loss_coefficient_calculator",
                                                                           value=1.0)
                for node in network.topologicalSortedNodes:
                    if node.isLeaf:
                        continue
                    # Probability Threshold
                    node_degree = GlobalConstants.TREE_DEGREE_LIST[node.depth]
                    initial_value = 1.0 / float(node_degree)
                    threshold_name = self.get_variable_name(name="prob_threshold_calculator", node=node)
                    # node.probThresholdCalculator = DecayingParameter(name=threshold_name, value=initial_value, decay=0.8,
                    #                                                  decay_period=70000,
                    #                                                  min_limit=0.4)
                    node.probThresholdCalculator = FixedParameter(name=threshold_name, value=initial_value)
                    # Softmax Decay
                    decay_name = self.get_variable_name(name="softmax_decay", node=node)
                    node.softmaxDecayCalculator = DecayingParameter(name=decay_name,
                                                                    value=GlobalConstants.RESNET_SOFTMAX_DECAY_INITIAL,
                                                                    decay=GlobalConstants.RESNET_SOFTMAX_DECAY_COEFFICIENT,
                                                                    decay_period=GlobalConstants.RESNET_SOFTMAX_DECAY_PERIOD,
                                                                    min_limit=GlobalConstants.RESNET_SOFTMAX_DECAY_MIN_LIMIT)
            self.decisionDropoutKeepProbCalculator = self.towerNetworks[0][1].decisionDropoutKeepProbCalculator
            self.noiseCoefficientCalculator = self.towerNetworks[0][1].noiseCoefficientCalculator
            self.decisionLossCoefficientCalculator = self.towerNetworks[0][1].decisionLossCoefficientCalculator

    def prepare_feed_dict(self, minibatch, iteration, use_threshold, is_train, use_masking):
        # Load the placeholders in each tower separately
        feed_dict = {self.iterationHolder: iteration}
        # Global parameters
        for tower_id, tpl in enumerate(self.towerNetworks):
            device_str = tpl[0]
            network = tpl[1]
            lower_bound = int(tower_id * self.towerBatchSize)
            upper_bound = int((tower_id + 1) * self.towerBatchSize)
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
            feed_dict[network.filteredMask] = np.ones((int(self.towerBatchSize),), dtype=bool)
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
        run_ops = [self.applyGradientsOp, self.learningRate, self.sampleCountTensors, self.isOpenTensors,
                   self.infoGainDicts]
        # run_ops.extend(self.batchNormMovingAvgAssignOps)
        # run_ops = [self.learningRate, self.sampleCountTensors, self.isOpenTensors,
        #            self.infoGainDicts]
        return run_ops

    def update_params(self, sess, dataset, epoch, iteration):
        use_threshold = int(GlobalConstants.USE_PROBABILITY_THRESHOLD)
        GlobalConstants.CURR_BATCH_SIZE = GlobalConstants.BATCH_SIZE
        minibatch = dataset.get_next_batch()
        if minibatch is None:
            return None, None, None
        feed_dict = self.prepare_feed_dict(minibatch=minibatch, iteration=iteration, use_threshold=use_threshold,
                                           is_train=True, use_masking=True)
        # Prepare result tensors to collect
        run_ops = self.get_run_ops()
        if GlobalConstants.USE_VERBOSE:
            run_ops.append(self.evalDict)
        results = sess.run(run_ops, feed_dict=feed_dict)
        self.unit_test_batch_norm_ops(sess=sess, eval_results=results[-1])
        lr = results[1]
        sample_counts = results[2]
        is_open_indicators = results[3]
        return lr, sample_counts, is_open_indicators

    def unit_test_batch_norm_ops(self, sess, eval_results):
        batch_norm_moving_averages = tf.get_collection(CustomBatchNormAlgorithms.BATCH_NORM_OPS)
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
            tf_moving_average_value = res[var_name]
            new_value_arrays = [np.expand_dims(v[1], axis=0) for v in values]
            unified_arr = np.concatenate(new_value_arrays, axis=0)
            mean_arr = np.mean(unified_arr, axis=0)
            if var_name not in self.batchNormMovingAverageValues:
                self.batchNormMovingAverageValues[var_name] = mean_arr
            else:
                curr_value = self.batchNormMovingAverageValues[var_name]
                self.batchNormMovingAverageValues[var_name] = momentum * curr_value + (1.0 - momentum) * mean_arr
            if not np.allclose(self.batchNormMovingAverageValues[var_name], tf_moving_average_value):
                print("X")

    # TODO: At sometime in the future, we should implement multi gpu accuracy calculation as well.
    #  But not now (31.05.2019)
    def calculate_accuracy(self, calculation_type, sess, dataset, dataset_type, run_id, iteration):
        network = self.towerNetworks[0][1]
        if not network.modeTracker.isCompressed:
            if calculation_type == AccuracyCalcType.regular:
                accuracy, confusion = network.accuracyCalculator.calculate_accuracy(sess=sess, dataset=dataset,
                                                                                    dataset_type=dataset_type,
                                                                                    run_id=run_id,
                                                                                    iteration=iteration)
                return accuracy, confusion
            elif calculation_type == AccuracyCalcType.route_correction:
                accuracy_corrected, marginal_corrected = \
                    network.accuracyCalculator.calculate_accuracy_with_route_correction(
                        sess=sess, dataset=dataset,
                        dataset_type=dataset_type)
                return accuracy_corrected, marginal_corrected
            elif calculation_type == AccuracyCalcType.with_residue_network:
                network.accuracyCalculator.calculate_accuracy_with_residue_network(sess=sess, dataset=dataset,
                                                                                   dataset_type=dataset_type)
            elif calculation_type == AccuracyCalcType.multi_path:
                network.accuracyCalculator.calculate_accuracy_multipath(sess=sess, dataset=dataset,
                                                                        dataset_type=dataset_type, run_id=run_id,
                                                                        iteration=iteration)
            else:
                raise NotImplementedError()
        else:
            best_leaf_accuracy, residue_corrected_accuracy = \
                network.accuracyCalculator.calculate_accuracy_after_compression(sess=sess, dataset=dataset,
                                                                                dataset_type=dataset_type,
                                                                                run_id=run_id, iteration=iteration)
            return best_leaf_accuracy, residue_corrected_accuracy
