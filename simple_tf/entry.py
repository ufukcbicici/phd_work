from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from auxillary.constants import DatasetTypes
from auxillary.dag_utilities import Dag

import argparse
import gzip
import os
import sys
import time
import networkx as nx

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import itertools

# MNIST
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from data_handling.mnist_data_set import MnistDataSet
from simple_tf.global_params import GlobalConstants
from simple_tf.tree import TreeNetwork
import simple_tf.lenet3 as lenet3
import simple_tf.baseline as baseline


# tf.set_random_seed(1234)
# np_seed = 88
# np.random.seed(np_seed)


def get_explanation_string(network):
    total_param_count = 0
    for v in tf.trainable_variables():
        total_param_count += np.prod(v.get_shape().as_list())

    # Tree
    explanation = "Tree.\n"
    explanation += "Batch Size:{0}\n".format(GlobalConstants.BATCH_SIZE)
    explanation += "Tree Degree:{0}\n".format(GlobalConstants.TREE_DEGREE_LIST)
    explanation += "Concat Trick:{0}\n".format(GlobalConstants.USE_CONCAT_TRICK)
    explanation += "Info Gain:{0}\n".format(GlobalConstants.USE_INFO_GAIN_DECISION)
    explanation += "Gradient Type:{0}\n".format(GlobalConstants.GRADIENT_TYPE)
    explanation += "Probability Threshold:{0}\n".format(GlobalConstants.USE_PROBABILITY_THRESHOLD)
    explanation += "Initial Lr:{0}\n".format(GlobalConstants.INITIAL_LR)
    explanation += "Decay Steps:{0}\n".format(GlobalConstants.DECAY_STEP)
    explanation += "Decay Rate:{0}\n".format(GlobalConstants.DECAY_RATE)
    explanation += "Param Count:{0}\n".format(total_param_count)
    explanation += "Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
    explanation += "Decision Wd:{0}\n".format(GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT)
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
    explanation += "Use Decision Dropout:{0}\n".format(GlobalConstants.USE_DROPOUT_FOR_DECISION)
    if GlobalConstants.USE_DROPOUT_FOR_DECISION:
        explanation += "********Decision Dropout Schedule********\n"
        explanation += "Iteration:{0} Probability:{1}\n".format(0, GlobalConstants.DROPOUT_INITIAL_PROB)
        for tpl in GlobalConstants.DROPOUT_SCHEDULE:
            explanation += "Iteration:{0} Probability:{1}\n".format(tpl[0], tpl[1])
        explanation += "********Decision Dropout Schedule********\n"
    explanation += "Use Classification Dropout:{0}\n".format(GlobalConstants.USE_DROPOUT_FOR_CLASSIFICATION)
    explanation += "Classification Dropout Probability:{0}\n".format(GlobalConstants.CLASSIFICATION_DROPOUT_PROB)
    if GlobalConstants.USE_PROBABILITY_THRESHOLD:
        for node in network.topologicalSortedNodes:
            if node.isLeaf:
                continue
            explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
            explanation += "Prob Threshold Initial Value:{0}\n".format(node.probThresholdCalculator.value)
            explanation += "Prob Threshold Decay Step:{0}\n".format(node.probThresholdCalculator.decayPeriod)
            explanation += "Prob Threshold Decay Ratio:{0}\n".format(node.probThresholdCalculator.decay)
            explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)

    # Baseline
    # explanation = "Baseline.\n"
    # explanation += "Batch Size:{0}\n".format(GlobalConstants.BATCH_SIZE)
    # explanation += "Gradient Type:{0}\n".format(GlobalConstants.GRADIENT_TYPE)
    # explanation += "Initial Lr:{0}\n".format(GlobalConstants.INITIAL_LR)
    # explanation += "Decay Steps:{0}\n".format(GlobalConstants.DECAY_STEP)
    # explanation += "Decay Rate:{0}\n".format(GlobalConstants.DECAY_RATE)
    # explanation += "Param Count:{0}\n".format(total_param_count)
    # explanation += "Model: {0}Conv - {1}Conv - {2}FC\n".format(GlobalConstants.NO_FILTERS_1, GlobalConstants.NO_FILTERS_2,
    #                                                          GlobalConstants.NO_HIDDEN)
    # explanation += "Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
    return explanation


def main():
    # Do the training
    if GlobalConstants.USE_CPU:
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()
    dataset = MnistDataSet(validation_sample_count=10000, load_validation_from="validation_indices")
    # Build the network
    # network = TreeNetwork(tree_degree=GlobalConstants.TREE_DEGREE,
    #                       node_build_funcs=[lenet3.root_func, lenet3.l1_func, lenet3.leaf_func],
    #                       grad_func=lenet3.grad_func,
    #                       create_new_variables=True)
    network = TreeNetwork(  # tree_degree=GlobalConstants.TREE_DEGREE,
        # node_build_funcs=[baseline.baseline],
        node_build_funcs=[lenet3.root_func, lenet3.l1_func, lenet3.leaf_func],
        grad_func=lenet3.grad_func,
        threshold_func=lenet3.threshold_calculator_func,
        summary_func=lenet3.tensorboard_func,
        degree_list=GlobalConstants.TREE_DEGREE_LIST)
    network.build_network()
    # dataset.reset()
    # Init
    init = tf.global_variables_initializer()
    # Grid search
    # wd_list = [0.0001 * x for n in range(0, 31) for x in itertools.repeat(n, 5)] # list(itertools.product(*list_of_lists))
    # wd_list = [x for x in itertools.repeat(0.0, 5)]
    cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[[0.0000375], [0.003]])
    # del cartesian_product[0:10]
    # wd_list = [0.02]
    run_id = 0
    for tpl in cartesian_product:
        print("********************NEW RUN:{0}********************".format(run_id))
        # Restart the network; including all annealed parameters.
        GlobalConstants.WEIGHT_DECAY_COEFFICIENT = tpl[0]
        GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT = tpl[1]
        # GlobalConstants.CLASSIFICATION_DROPOUT_PROB = tpl[2]
        network.thresholdFunc(network=network)
        experiment_id = DbLogger.get_run_id()
        explanation = get_explanation_string(network=network)
        DbLogger.write_into_table(rows=[(experiment_id, explanation)], table=DbLogger.runMetaData,
                                  col_count=2)
        sess.run(init)
        iteration_counter = 0
        for epoch_id in range(GlobalConstants.EPOCH_COUNT):
            # An epoch is a complete pass on the whole dataset.
            dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
            print("*************Epoch {0}*************".format(epoch_id))
            total_time = 0.0
            leaf_info_rows = []
            while True:
                start_time = time.time()
                sample_counts, lr, is_open_indicators = network.update_params_with_momentum(sess=sess, dataset=dataset,
                                                                                            iteration=iteration_counter)
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                print("Iteration:{0}".format(iteration_counter))
                print("Lr:{0}".format(lr))
                # Print sample counts
                sample_count_str = ""
                for k, v in sample_counts.items():
                    sample_count_str += "[{0}={1}]".format(k, v)
                    node_index = network.get_node_from_variable_name(name=k).index
                    leaf_info_rows.append((node_index, np.asscalar(v), iteration_counter, experiment_id))
                indicator_str = ""
                for k, v in is_open_indicators.items():
                    indicator_str += "[{0}={1}]".format(k, v)
                print(sample_count_str)
                print(indicator_str)
                iteration_counter += 1
                if iteration_counter % 50 == 0:
                    kv_rows = []
                    for k, v in sample_counts.items():
                        kv_rows.append((experiment_id, iteration_counter, k, np.asscalar(v)))
                    DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore, col_count=4)
                if dataset.isNewEpoch:
                    if (epoch_id + 1) % GlobalConstants.EPOCH_REPORT_PERIOD == 0:
                        print("Epoch Time={0}".format(total_time))
                        training_accuracy, training_confusion = \
                            network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.training,
                                                       run_id=experiment_id, iteration=iteration_counter)
                        validation_accuracy, validation_confusion = \
                            network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.validation,
                                                       run_id=experiment_id, iteration=iteration_counter)
                        validation_accuracy_corrected = \
                            network.calculate_accuracy_with_route_correction(sess=sess, dataset=dataset,
                                                                             dataset_type=DatasetTypes.validation,
                                                                             run_id=experiment_id,
                                                                             iteration=iteration_counter)
                        DbLogger.write_into_table(rows=[(experiment_id, iteration_counter,
                                                         "Corrected Validation Accuracy",
                                                         validation_accuracy_corrected)],
                                                  table=DbLogger.runKvStore, col_count=4)
                        DbLogger.write_into_table(rows=[(experiment_id, iteration_counter, epoch_id, training_accuracy,
                                                         validation_accuracy,
                                                         0.0, 0.0, "LeNet3")], table=DbLogger.logsTable, col_count=8)
                        DbLogger.write_into_table(rows=leaf_info_rows, table=DbLogger.leafInfoTable, col_count=4)
                        if GlobalConstants.SAVE_CONFUSION_MATRICES:
                            DbLogger.write_into_table(rows=training_confusion, table=DbLogger.confusionTable,
                                                      col_count=7)
                            DbLogger.write_into_table(rows=validation_confusion, table=DbLogger.confusionTable,
                                                      col_count=7)
                        leaf_info_rows = []
                    break
        test_accuracy, test_confusion = network.calculate_accuracy(sess=sess, dataset=dataset,
                                                                   dataset_type=DatasetTypes.test,
                                                                   run_id=experiment_id, iteration=iteration_counter)
        test_accuracy_corrected = \
            network.calculate_accuracy_with_route_correction(sess=sess, dataset=dataset,
                                                             dataset_type=DatasetTypes.test,
                                                             run_id=experiment_id,
                                                             iteration=iteration_counter)
        DbLogger.write_into_table(rows=[(experiment_id, iteration_counter,
                                         "Corrected Test Accuracy",
                                         test_accuracy_corrected)],
                                  table=DbLogger.runKvStore, col_count=4)
        DbLogger.write_into_table([(experiment_id, explanation, test_accuracy)], table=DbLogger.runResultsTable,
                                  col_count=3)
        if GlobalConstants.SAVE_CONFUSION_MATRICES:
            DbLogger.write_into_table(rows=test_confusion, table=DbLogger.confusionTable, col_count=7)
        print("X")
        run_id += 1


main()
