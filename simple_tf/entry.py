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
from auxillary.parameters import DecayingParameterV2, DecayingParameter
from data_handling.fashion_mnist import FashionMnistDataSet
from data_handling.mnist_data_set import MnistDataSet
from simple_tf import lenet_decision_connected_to_f, fashion_net_baseline
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
    # explanation = "Tree H Connected to F, With Dropout in H. Run 1\n"
    # explanation += "Batch Size:{0}\n".format(GlobalConstants.BATCH_SIZE)
    # explanation += "Tree Degree:{0}\n".format(GlobalConstants.TREE_DEGREE_LIST)
    # explanation += "Concat Trick:{0}\n".format(GlobalConstants.USE_CONCAT_TRICK)
    # explanation += "Info Gain:{0}\n".format(GlobalConstants.USE_INFO_GAIN_DECISION)
    # explanation += "Gradient Type:{0}\n".format(GlobalConstants.GRADIENT_TYPE)
    # explanation += "Probability Threshold:{0}\n".format(GlobalConstants.USE_PROBABILITY_THRESHOLD)
    # explanation += "Initial Lr:{0}\n".format(GlobalConstants.INITIAL_LR)
    # explanation += "Decay Steps:{0}\n".format(GlobalConstants.DECAY_STEP)
    # explanation += "Decay Rate:{0}\n".format(GlobalConstants.DECAY_RATE)
    # explanation += "Param Count:{0}\n".format(total_param_count)
    # explanation += "Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
    # explanation += "Decision Wd:{0}\n".format(GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT)
    # explanation += "Using Info Gain:{0}\n".format(GlobalConstants.USE_INFO_GAIN_DECISION)
    # explanation += "Info Gain Loss Lambda:{0}\n".format(GlobalConstants.DECISION_LOSS_COEFFICIENT)
    # explanation += "Use Batch Norm Before Decisions:{0}\n".format(GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING)
    # explanation += "Use Trainable Batch Norm Parameters:{0}\n".format(
    #     GlobalConstants.USE_TRAINABLE_PARAMS_WITH_BATCH_NORM)
    # explanation += "Hyperplane bias at 0.0\n"
    # explanation += "Using Convolutional Routing Networks:{0}\n".format(GlobalConstants.USE_CONVOLUTIONAL_H_PIPELINE)
    # explanation += "Softmax Decay Initial:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_INITIAL)
    # explanation += "Softmax Decay Coefficient:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_COEFFICIENT)
    # explanation += "Softmax Decay Period:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_PERIOD)
    # explanation += "Softmax Min Limit:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_MIN_LIMIT)
    # explanation += "Reparametrized Noise:{0}\n".format(GlobalConstants.USE_REPARAMETRIZATION_TRICK)
    # explanation += "Info Gain Balance Coefficient:{0}\n".format(GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT)
    # explanation += "Adaptive Weight Decay:{0}\n".format(GlobalConstants.USE_ADAPTIVE_WEIGHT_DECAY)
    # if GlobalConstants.USE_REPARAMETRIZATION_TRICK:
    #     explanation += "********Reparametrized Noise Settings********\n"
    #     explanation += "Noise Coefficient Initial Value:{0}\n".format(network.noiseCoefficientCalculator.value)
    #     explanation += "Noise Coefficient Decay Step:{0}\n".format(network.noiseCoefficientCalculator.decayPeriod)
    #     explanation += "Noise Coefficient Decay Ratio:{0}\n".format(network.noiseCoefficientCalculator.decay)
    #     explanation += "********Reparametrized Noise Settings********\n"
    # explanation += "Use Decision Dropout:{0}\n".format(GlobalConstants.USE_DROPOUT_FOR_DECISION)
    # explanation += "Use Decision Augmentation:{0}\n".format(GlobalConstants.USE_DECISION_AUGMENTATION)
    # if GlobalConstants.USE_DROPOUT_FOR_DECISION:
    #     explanation += "********Decision Dropout Schedule********\n"
    #     explanation += "Iteration:{0} Probability:{1}\n".format(0, GlobalConstants.DROPOUT_INITIAL_PROB)
    #     for tpl in GlobalConstants.DROPOUT_SCHEDULE:
    #         explanation += "Iteration:{0} Probability:{1}\n".format(tpl[0], tpl[1])
    #     explanation += "********Decision Dropout Schedule********\n"
    # explanation += "Use Classification Dropout:{0}\n".format(GlobalConstants.USE_DROPOUT_FOR_CLASSIFICATION)
    # explanation += "Classification Dropout Probability:{0}\n".format(GlobalConstants.CLASSIFICATION_DROPOUT_PROB)
    # if GlobalConstants.USE_PROBABILITY_THRESHOLD:
    #     for node in network.topologicalSortedNodes:
    #         if node.isLeaf:
    #             continue
    #         explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
    #         explanation += "Prob Threshold Initial Value:{0}\n".format(node.probThresholdCalculator.value)
    #         explanation += "Prob Threshold Decay Step:{0}\n".format(node.probThresholdCalculator.decayPeriod)
    #         explanation += "Prob Threshold Decay Ratio:{0}\n".format(node.probThresholdCalculator.decay)
    #         explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)

    # Baseline
    explanation = "Fashion Mnist Baseline. Double Dropout Conv Filters:5x5 - 5x5 - 1x1." \
                  "(Lr=0.01, - Decay 0.1 at each 15000. iteration)\n"
    explanation += "Batch Size:{0}\n".format(GlobalConstants.BATCH_SIZE)
    explanation += "Gradient Type:{0}\n".format(GlobalConstants.GRADIENT_TYPE)
    explanation += "Initial Lr:{0}\n".format(GlobalConstants.INITIAL_LR)
    explanation += "Decay Steps:{0}\n".format(GlobalConstants.DECAY_STEP)
    explanation += "Decay Rate:{0}\n".format(GlobalConstants.DECAY_RATE)
    explanation += "Param Count:{0}\n".format(total_param_count)
    explanation += "Model: {0}Conv - {1}Conv - {2}Conv - {3}FC - {4}FC\n". \
        format(GlobalConstants.FASHION_NUM_FILTERS_1, GlobalConstants.FASHION_NUM_FILTERS_2,
               GlobalConstants.FASHION_NUM_FILTERS_3, GlobalConstants.FASHION_FC_1, GlobalConstants.FASHION_FC_2)
    explanation += "Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
    explanation += "Dropout Prob:{0}\n".format(GlobalConstants.CLASSIFICATION_DROPOUT_PROB)
    return explanation


def main():
    # Do the training
    if GlobalConstants.USE_CPU:
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()
        dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
    # Build the network
    # network = TreeNetwork(tree_degree=GlobalConstants.TREE_DEGREE,
    #                       node_build_funcs=[lenet3.root_func, lenet3.l1_func, lenet3.leaf_func],
    #                       grad_func=lenet3.grad_func,
    #                       create_new_variables=True)
    network = TreeNetwork(  # tree_degree=GlobalConstants.TREE_DEGREE,
        # node_build_funcs=[baseline.baseline],
        # node_build_funcs=[lenet_decision_connected_to_f.root_func, lenet_decision_connected_to_f.l1_func,
        #                   lenet_decision_connected_to_f.leaf_func],
        node_build_funcs=[fashion_net_baseline.baseline],
        grad_func=fashion_net_baseline.grad_func,
        threshold_func=fashion_net_baseline.threshold_calculator_func,
        residue_func=fashion_net_baseline.residue_network_func,
        summary_func=fashion_net_baseline.tensorboard_func,
        degree_list=GlobalConstants.TREE_DEGREE_LIST)
    network.build_network()
    # dataset.reset()
    # Init
    init = tf.global_variables_initializer()
    # Grid search
    # wd_list = [0.0001 * x for n in range(0, 31) for x in itertools.repeat(n, 5)] # list(itertools.product(*list_of_lists))
    # # wd_list = [x for x in itertools.repeat(0.0, 5)]
    # cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[[0.0000375, 0.0000375, 0.0000375,
    #                                                                        0.00005, 0.00005, 0.00005,
    #                                                                        0.000075, 0.000075, 0.000075,
    #                                                                        0.0001, 0.0001, 0.0001,
    #                                                                        0.000125, 0.000125, 0.000125], [0.0009]])
    # classification_wd = [0.00005 * x for n in range(0, 16) for x in itertools.repeat(n, 3)]
    # decision_wd = [0.0]
    cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[[0.0, 0.0, 0.0, 0.0, 0.0],
                                                                          [0.0]])
    # del cartesian_product[0:10]
    # wd_list = [0.02]
    run_id = 0
    for tpl in cartesian_product:
        print("********************NEW RUN:{0}********************".format(run_id))
        # Restart the network; including all annealed parameters.
        GlobalConstants.WEIGHT_DECAY_COEFFICIENT = tpl[0]
        GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT = tpl[1]
        GlobalConstants.LEARNING_RATE_CALCULATOR = DecayingParameter(name="lr_calculator",
                                                                     value=GlobalConstants.INITIAL_LR,
                                                                     decay=GlobalConstants.DECAY_RATE,
                                                                     decay_period=GlobalConstants.DECAY_STEP)
        network.learningRateCalculator = GlobalConstants.LEARNING_RATE_CALCULATOR
        # GlobalConstants.LEARNING_RATE_CALCULATOR = DecayingParameterV2(name="lr_calculator",
        #                                                                value=GlobalConstants.INITIAL_LR,
        #                                                                decay=GlobalConstants.DECAY_RATE)
        # GlobalConstants.CLASSIFICATION_DROPOUT_PROB = tpl[2]
        network.thresholdFunc(network=network)
        experiment_id = DbLogger.get_run_id()
        explanation = get_explanation_string(network=network)
        DbLogger.write_into_table(rows=[(experiment_id, explanation)], table=DbLogger.runMetaData,
                                  col_count=2)
        sess.run(init)
        iteration_counter = 0
        for epoch_id in range(GlobalConstants.TOTAL_EPOCH_COUNT):
            # An epoch is a complete pass on the whole dataset.
            dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
            print("*************Epoch {0}*************".format(epoch_id))
            total_time = 0.0
            leaf_info_rows = []
            while True:
                start_time = time.time()
                sample_counts, decision_sample_counts, lr, is_open_indicators = \
                    network.update_params_with_momentum(sess=sess, dataset=dataset,
                                                        epoch=epoch_id,
                                                        iteration=iteration_counter)
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                print("Iteration:{0}".format(iteration_counter))
                print("Lr:{0}".format(lr))
                # Print sample counts (decision)
                if decision_sample_counts is not None:
                    decision_sample_count_str = "Decision:   "
                    for k, v in decision_sample_counts.items():
                        decision_sample_count_str += "[{0}={1}]".format(k, v)
                        node_index = network.get_node_from_variable_name(name=k).index
                    # leaf_info_rows.append((node_index, np.asscalar(v), iteration_counter, experiment_id))
                    print(decision_sample_count_str)
                # Print sample counts (classification)
                sample_count_str = "Classification:   "
                for k, v in sample_counts.items():
                    sample_count_str += "[{0}={1}]".format(k, v)
                    node_index = network.get_node_from_variable_name(name=k).index
                    leaf_info_rows.append((node_index, np.asscalar(v), iteration_counter, experiment_id))
                print(sample_count_str)
                indicator_str = ""
                for k, v in is_open_indicators.items():
                    indicator_str += "[{0}={1}]".format(k, v)
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
                            network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.test,
                                                       run_id=experiment_id, iteration=iteration_counter)
                        if not network.isBaseline:
                            validation_accuracy_corrected = \
                                network.calculate_accuracy_with_route_correction(sess=sess, dataset=dataset,
                                                                                 dataset_type=DatasetTypes.test,
                                                                                 run_id=experiment_id,
                                                                                 iteration=iteration_counter)
                        # network.calculate_accuracy_with_residue_network(sess=sess, dataset=dataset,
                        #                                                 dataset_type=DatasetTypes.validation)
                        # DbLogger.write_into_table(rows=[(experiment_id, iteration_counter,
                        #                                  "Corrected Validation Accuracy",
                        #                                  validation_accuracy_corrected)],
                        #                           table=DbLogger.runKvStore, col_count=4)
                        DbLogger.write_into_table(rows=[(experiment_id, iteration_counter, epoch_id, training_accuracy,
                                                         validation_accuracy, validation_accuracy,
                                                         0.0, 0.0, "LeNet3")], table=DbLogger.logsTable, col_count=9)
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
        result_rows = [(experiment_id, explanation, test_accuracy, "Regular")]
        if not network.isBaseline:
            test_accuracy_corrected = \
                network.calculate_accuracy_with_route_correction(sess=sess, dataset=dataset,
                                                                 dataset_type=DatasetTypes.test,
                                                                 run_id=experiment_id,
                                                                 iteration=iteration_counter)
            network.calculate_accuracy_with_residue_network(sess=sess, dataset=dataset, dataset_type=DatasetTypes.test)
            DbLogger.write_into_table(rows=[(experiment_id, iteration_counter,
                                             "Corrected Test Accuracy",
                                             test_accuracy_corrected)],
                                      table=DbLogger.runKvStore, col_count=4)
            result_rows.append((experiment_id, explanation, test_accuracy_corrected, "Corrected"))
        DbLogger.write_into_table(result_rows, table=DbLogger.runResultsTable, col_count=4)
        if GlobalConstants.SAVE_CONFUSION_MATRICES:
            DbLogger.write_into_table(rows=test_confusion, table=DbLogger.confusionTable, col_count=7)
        print("X")
        run_id += 1


main()
