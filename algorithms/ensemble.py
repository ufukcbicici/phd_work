import time

from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import FixedParameter
from data_handling.data_set import DataSet
from simple_tf.global_params import GlobalConstants
import numpy as np
import tensorflow as tf


def get_explanation_string(networks):
    total_param_count = 0
    for network in networks:
        for v in network.variableManager.trainable_variables():
            total_param_count += np.prod(v.get_shape().as_list())

    # Tree
    explanation = "FashionMnist Ensemble of Thin Baselines \n"
    # "(Lr=0.01, - Decay 1/(1 + i*0.0001) at each i. iteration)\n"
    explanation += "Using Fast Tree Version:{0}\n".format(GlobalConstants.USE_FAST_TREE_MODE)
    explanation += "Batch Size:{0}\n".format(GlobalConstants.BATCH_SIZE)
    explanation += "Tree Degree:{0}\n".format(GlobalConstants.TREE_DEGREE_LIST)
    explanation += "Concat Trick:{0}\n".format(GlobalConstants.USE_CONCAT_TRICK)
    explanation += "Ensemble Count:{0}\n".format(GlobalConstants.BASELINE_ENSEMBLE_COUNT)
    explanation += "Info Gain:{0}\n".format(GlobalConstants.USE_INFO_GAIN_DECISION)
    explanation += "Using Effective Sample Counts:{0}\n".format(GlobalConstants.USE_EFFECTIVE_SAMPLE_COUNTS)
    explanation += "Gradient Type:{0}\n".format(GlobalConstants.GRADIENT_TYPE)
    explanation += "Probability Threshold:{0}\n".format(GlobalConstants.USE_PROBABILITY_THRESHOLD)
    explanation += "********Lr Settings********\n"
    explanation += GlobalConstants.LEARNING_RATE_CALCULATOR.get_explanation()
    explanation += "********Lr Settings********\n"
    explanation += "Use Unified Batch Norm:{0}\n".format(GlobalConstants.USE_UNIFIED_BATCH_NORM)
    explanation += "Batch Norm Decay:{0}\n".format(GlobalConstants.BATCH_NORM_DECAY)
    explanation += "Param Count:{0}\n".format(total_param_count)
    explanation += "Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
    explanation += "Decision Wd:{0}\n".format(GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT)
    explanation += "Residue Loss Coefficient:{0}\n".format(GlobalConstants.RESIDUE_LOSS_COEFFICIENT)
    explanation += "Residue Affects All Network:{0}\n".format(GlobalConstants.RESIDE_AFFECTS_WHOLE_NETWORK)
    explanation += "Using Info Gain:{0}\n".format(GlobalConstants.USE_INFO_GAIN_DECISION)
    explanation += "Info Gain Loss Lambda:{0}\n".format(GlobalConstants.DECISION_LOSS_COEFFICIENT)
    explanation += "Use Batch Norm Before Decisions:{0}\n".format(GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING)
    explanation += "Use Trainable Batch Norm Parameters:{0}\n".format(
        GlobalConstants.USE_TRAINABLE_PARAMS_WITH_BATCH_NORM)
    explanation += "Hyperplane bias at 0.0\n"
    explanation += "Using Convolutional Routing Networks:{0}\n".format(GlobalConstants.USE_CONVOLUTIONAL_H_PIPELINE)
    explanation += "Softmax Decay Initial:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_INITIAL)
    explanation += "Softmax Decay Coefficient:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_COEFFICIENT)
    explanation += "Softmax Decay Period:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_PERIOD)
    explanation += "Softmax Min Limit:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_MIN_LIMIT)
    explanation += "Softmax Test Temperature:{0}\n".format(GlobalConstants.SOFTMAX_TEST_TEMPERATURE)
    explanation += "Reparametrized Noise:{0}\n".format(GlobalConstants.USE_REPARAMETRIZATION_TRICK)
    # for node in network.topologicalSortedNodes:
    #     if node.isLeaf:
    #         continue
    #     explanation += "Node {0} Info Gain Balance Coefficient:{1}\n".format(node.index,
    #                                                                          node.infoGainBalanceCoefficient)
    explanation += "Info Gain Balance Coefficient:{0}\n".format(GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT)
    explanation += "Adaptive Weight Decay:{0}\n".format(GlobalConstants.USE_ADAPTIVE_WEIGHT_DECAY)
    if GlobalConstants.USE_REPARAMETRIZATION_TRICK:
        explanation += "********Reparametrized Noise Settings********\n"
        explanation += "Noise Coefficient Initial Value:{0}\n".format(networks[0].noiseCoefficientCalculator.value)
        explanation += "Noise Coefficient Decay Step:{0}\n".format(networks[0].noiseCoefficientCalculator.decayPeriod)
        explanation += "Noise Coefficient Decay Ratio:{0}\n".format(networks[0].noiseCoefficientCalculator.decay)
        explanation += "********Reparametrized Noise Settings********\n"
    explanation += "Use Decision Dropout:{0}\n".format(GlobalConstants.USE_DROPOUT_FOR_DECISION)
    explanation += "Use Decision Augmentation:{0}\n".format(GlobalConstants.USE_DECISION_AUGMENTATION)
    if GlobalConstants.USE_DROPOUT_FOR_DECISION:
        explanation += "********Decision Dropout Schedule********\n"
        explanation += "Iteration:{0} Probability:{1}\n".format(0, GlobalConstants.DROPOUT_INITIAL_PROB)
        for tpl in GlobalConstants.DROPOUT_SCHEDULE:
            explanation += "Iteration:{0} Probability:{1}\n".format(tpl[0], tpl[1])
        explanation += "********Decision Dropout Schedule********\n"
    explanation += "Use Classification Dropout:{0}\n".format(GlobalConstants.USE_DROPOUT_FOR_CLASSIFICATION)
    explanation += "Classification Dropout Probability:{0}\n".format(GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB)
    explanation += "Decision Dropout Probability:{0}\n".format(networks[0].decisionDropoutKeepProbCalculator.value)
    if GlobalConstants.USE_PROBABILITY_THRESHOLD:
        for node in networks[0].topologicalSortedNodes:
            if node.isLeaf:
                continue
            explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
            explanation += node.probThresholdCalculator.get_explanation()
            explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
    explanation += "Use Softmax Compression:{0}\n".format(GlobalConstants.USE_SOFTMAX_DISTILLATION)
    explanation += "Waiting Epochs for Softmax Compression:{0}\n".format(GlobalConstants.MODE_WAIT_EPOCHS)
    explanation += "Mode Percentile:{0}\n".format(GlobalConstants.PERCENTILE_THRESHOLD)
    explanation += "Mode Tracking Strategy:{0}\n".format(GlobalConstants.MODE_TRACKING_STRATEGY)
    explanation += "Mode Max Class Count:{0}\n".format(GlobalConstants.MAX_MODE_CLASSES)
    explanation += "Mode Computation Strategy:{0}\n".format(GlobalConstants.MODE_COMPUTATION_STRATEGY)
    explanation += "Constrain Softmax Compression With Label Count:{0}\n".format(GlobalConstants.
                                                                                 CONSTRAIN_WITH_COMPRESSION_LABEL_COUNT)
    explanation += "Softmax Distillation Cross Validation Count:{0}\n". \
        format(GlobalConstants.SOFTMAX_DISTILLATION_CROSS_VALIDATION_COUNT)
    explanation += "Softmax Distillation Strategy:{0}\n". \
        format(GlobalConstants.SOFTMAX_COMPRESSION_STRATEGY)
    explanation += "F Conv1:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_FILTERS_1_SIZE,
                                                           GlobalConstants.FASHION_F_NUM_FILTERS_1)
    explanation += "F Conv2:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_FILTERS_2_SIZE,
                                                           GlobalConstants.FASHION_F_NUM_FILTERS_2)
    explanation += "F Conv3:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_FILTERS_3_SIZE,
                                                           GlobalConstants.FASHION_F_NUM_FILTERS_3)
    explanation += "F FC1:{0} Units\n".format(GlobalConstants.FASHION_F_FC_1)
    explanation += "F FC2:{0} Units\n".format(GlobalConstants.FASHION_F_FC_2)
    explanation += "F Residue FC:{0} Units\n".format(GlobalConstants.FASHION_F_RESIDUE)
    explanation += "Residue Hidden Layer Count:{0}\n".format(GlobalConstants.FASHION_F_RESIDUE_LAYER_COUNT)
    explanation += "Residue Use Dropout:{0}\n".format(GlobalConstants.FASHION_F_RESIDUE_USE_DROPOUT)
    explanation += "H Conv1:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_H_FILTERS_1_SIZE,
                                                           GlobalConstants.FASHION_H_NUM_FILTERS_1)
    explanation += "H Conv2:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_H_FILTERS_2_SIZE,
                                                           GlobalConstants.FASHION_H_NUM_FILTERS_2)
    explanation += "FASHION_NO_H_FROM_F_UNITS_1:{0} Units\n".format(GlobalConstants.FASHION_NO_H_FROM_F_UNITS_1)
    explanation += "FASHION_NO_H_FROM_F_UNITS_2:{0} Units\n".format(GlobalConstants.FASHION_NO_H_FROM_F_UNITS_2)
    return explanation


class Ensemble:
    def __init__(self, networks, datasets):
        self.networks = networks
        self.datasets = datasets
        self.globalCounter = None
        self.learningRate = None
        self.extra_update_ops = None
        self.optimizer = None

    def update(self, sess, iteration):
        GlobalConstants.CURR_BATCH_SIZE = GlobalConstants.BATCH_SIZE
        minibatches = []
        use_threshold = int(GlobalConstants.USE_PROBABILITY_THRESHOLD)
        feed_dict = {}
        for network, dataset in zip(self.networks, self.datasets):
            minibatch = dataset.get_next_batch()
            minibatch = DataSet.MiniBatch(np.expand_dims(minibatch.samples, axis=3), minibatch.labels,
                                          minibatch.indices, minibatch.one_hot_labels, minibatch.hash_codes)
            minibatches.append(minibatch)
            feed_dict_temp = network.prepare_feed_dict(minibatch=minibatch, iteration=iteration,
                                                       use_threshold=use_threshold,
                                                       is_train=True, use_masking=True)
            total_size = len(feed_dict_temp) + len(feed_dict)
            feed_dict.update(feed_dict_temp)
            assert total_size == len(feed_dict)
        # Prepare result tensors to collect
        run_ops = [self.optimizer, self.learningRate]
        # if GlobalConstants.USE_VERBOSE:
        #     run_ops.append(self.evalDict)
        results = sess.run(run_ops, feed_dict=feed_dict)
        return results[1]

    def eval(self, sess, dataset, use_masking):
        GlobalConstants.CURR_BATCH_SIZE = GlobalConstants.EVAL_BATCH_SIZE
        minibatch = dataset.get_next_batch()
        minibatch = DataSet.MiniBatch(np.expand_dims(minibatch.samples, axis=3), minibatch.labels,
                                      minibatch.indices, minibatch.one_hot_labels, minibatch.hash_codes)
        feed_dict = {}
        for network in self.networks:
            feed_dict_temp = network.prepare_feed_dict(minibatch=minibatch, iteration=1000000, use_threshold=False,
                                                  is_train=False, use_masking=use_masking)
            total_size = len(feed_dict_temp) + len(feed_dict)
            feed_dict.update(feed_dict_temp)
            assert total_size == len(feed_dict)
        list_of_eval_dicts = [network.evalDict for network in self.networks]
        # g = tf.get_default_graph()
        # run_metadata = tf.RunMetadata()
        # results = sess.run(list_of_eval_dicts, feed_dict,
        #                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        #                    run_metadata=run_metadata)
        # opts = tf.profiler.ProfileOptionBuilder.float_operation()
        # flops = tf.profiler.profile(g, run_meta=run_metadata, cmd='op', options=opts)
        results = sess.run(list_of_eval_dicts, feed_dict)
        return results

    def train(self, sess, run_id, weight_decay_coeff, decision_weight_decay_coeff, info_gain_balance_coeff,
              classification_dropout_prob, decision_dropout_prob):
        GlobalConstants.WEIGHT_DECAY_COEFFICIENT = weight_decay_coeff
        GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT = decision_weight_decay_coeff
        GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT = info_gain_balance_coeff
        GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB = 1.0 - classification_dropout_prob
        for network in self.networks:
            network.build_network()
            network.decisionDropoutKeepProbCalculator = FixedParameter(name="decision_dropout_prob",
                                                                       value=1.0 - decision_dropout_prob)
            network.learningRateCalculator = GlobalConstants.LEARNING_RATE_CALCULATOR
        experiment_id = DbLogger.get_run_id()
        explanation = get_explanation_string(networks=self.networks)
        series_id = int(run_id / 15)
        explanation += "\n Series:{0}".format(series_id)
        DbLogger.write_into_table(rows=[(experiment_id, explanation)], table=DbLogger.runMetaData,
                                  col_count=2)
        network_dataset_pairs = zip(self.networks, self.datasets)
        for network, dataset in network_dataset_pairs:
            network.reset_network(dataset=dataset, run_id=experiment_id)
        # Combine the losses of each network and build an optimizer
        network_losses = [network.finalLoss for network in self.networks]
        ensemble_loss = tf.add_n(inputs=[network_losses])

        # Build optimizer
        self.globalCounter = tf.Variable(0, trainable=False)
        boundaries = [tpl[0] for tpl in GlobalConstants.LEARNING_RATE_CALCULATOR.schedule]
        values = [GlobalConstants.INITIAL_LR]
        values.extend([tpl[1] for tpl in GlobalConstants.LEARNING_RATE_CALCULATOR.schedule])
        self.learningRate = tf.train.piecewise_constant(self.globalCounter, boundaries, values)
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.extra_update_ops):
            self.optimizer = tf.train.MomentumOptimizer(self.learningRate, 0.9).minimize(ensemble_loss,
                                                                                         global_step=self.globalCounter)
        init = tf.global_variables_initializer()
        sess.run(init)
        iteration_counter = 0
        for epoch_id in range(GlobalConstants.TOTAL_EPOCH_COUNT):
            # An epoch is a complete pass on the whole dataset.
            for dataset in self.datasets:
                dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
            print("*************Epoch {0}*************".format(epoch_id))
            total_time = 0.0
            leaf_info_rows = []
            while True:
                start_time = time.time()
                lr = self.update(sess=sess, iteration=iteration_counter)
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                print("Iteration:{0}".format(iteration_counter))
                print("Lr:{0}".format(lr))
                iteration_counter += 1
                if all([dataset.isNewEpoch for dataset in self.datasets]):
                    training_accuracy = self.calculate_accuracy(sess=sess, dataset_type=DatasetTypes.training)
                    test_accuracy = self.calculate_accuracy(sess=sess, dataset_type=DatasetTypes.test)
                    print("Elapsed Time:{0}".format(total_time))
                    DbLogger.write_into_table(
                        rows=[(experiment_id, iteration_counter, epoch_id, training_accuracy,
                               test_accuracy, 0.0,
                               0.0, 0.0, "XXX")], table=DbLogger.logsTable, col_count=9)
                    break

    def calculate_accuracy(self, sess, dataset_type):
        leaf_posteriors_dict = {}
        leaf_true_labels_dict = {}
        self.datasets[0].set_current_data_set_type(dataset_type=dataset_type)
        while True:
            results = self.eval(sess=sess, dataset=self.datasets[0], use_masking=True)
            batch_sample_count = 0.0
            for network, eval_dict in zip(self.networks, results):
                for node in network.topologicalSortedNodes:
                    if node.isLeaf:
                        posterior_probs = eval_dict[network.get_variable_name(name="posterior_probs", node=node)]
                        true_labels = eval_dict["Node{0}_label_tensor".format(node.index)]
                        UtilityFuncs.concat_to_np_array_dict(dct=leaf_posteriors_dict, key=(network, node.index),
                                                             array=posterior_probs)
                        UtilityFuncs.concat_to_np_array_dict(dct=leaf_true_labels_dict, key=(network, node.index),
                                                             array=true_labels)
            # batch_sample_count += list(leaf_true_labels_dict.values())[0].shape[0]
            # if batch_sample_count != GlobalConstants.EVAL_BATCH_SIZE:
            #     raise Exception("Incorrect batch size:{0}".format(batch_sample_count))
            if self.datasets[0].isNewEpoch:
                break
        ensemble_length = len(leaf_posteriors_dict)
        posterior_tensor = np.stack(list(leaf_posteriors_dict.values()), axis=0)
        assert posterior_tensor.shape[0] == len(self.networks)
        averaged_posterior = np.mean(posterior_tensor, axis=0)
        assert averaged_posterior.shape[0] == self.datasets[0].get_current_sample_count()
        assert averaged_posterior.shape[1] == self.datasets[0].get_label_count()
        true_labels_list = list(leaf_true_labels_dict.values())
        for i in range(len(true_labels_list)-1):
            assert np.array_equal(true_labels_list[i], true_labels_list[i+1])
        true_label_arr = true_labels_list[0]
        total_correct = 0.0
        total_count = float(true_label_arr.shape[0])
        assert averaged_posterior.shape[0] == true_label_arr.shape[0]
        for i in range(averaged_posterior.shape[0]):
            sample_posterior = averaged_posterior[i, :]
            predicted_label = np.argmax(sample_posterior)
            true_label = true_label_arr[i]
            if predicted_label == true_label:
                total_correct += 1.0
        print("*************Overall {0} samples. Overall Accuracy:{1}*************"
              .format(total_count, total_correct / total_count))
        return total_correct / total_count
