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
from data_handling.mnist_data_set import MnistDataSet
from simple_tf.global_params import GlobalConstants
from simple_tf.tree import TreeNetwork
import simple_tf.lenet3 as lenet3
import simple_tf.baseline as baseline

tf.set_random_seed(1234)
np_seed = 88
np.random.seed(np_seed)


def get_explanation_string():
    total_param_count = 0
    for v in tf.trainable_variables():
        total_param_count += np.prod(v.get_shape().as_list())
    explanation = "Tree.\n"
    explanation += "Tree Degree:{0}\n".format(GlobalConstants.TREE_DEGREE)
    explanation += "Gradient Type:{0}\n".format(GlobalConstants.GRADIENT_TYPE)
    explanation += "Probability Threshold:{0}\n".format(GlobalConstants.USE_PROBABILITY_THRESHOLD)
    explanation += "Initial Lr:{0}\n".format(GlobalConstants.INITIAL_LR)
    explanation += "Decay Steps:{0}\n".format(GlobalConstants.DECAY_STEP)
    explanation += "Decay Rate:{0}\n".format(GlobalConstants.DECAY_RATE)
    explanation += "Param Count:{0}\n".format(total_param_count)
    explanation += "Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
    explanation += "Using Info Gain:{0}\n".format(GlobalConstants.USE_INFO_GAIN_DECISION)
    explanation += "Info Gain Loss Lambda:{0}\n".format(GlobalConstants.DECISION_LOSS_COEFFICIENT)
    if GlobalConstants.USE_PROBABILITY_THRESHOLD:
        explanation += "Prob Threshold Initial Value:{0}\n".format(GlobalConstants.PROBABILITY_THRESHOLD.value)
        explanation += "Prob Threshold Decay Step:{0}\n".format(GlobalConstants.PROBABILITY_THRESHOLD.decayPeriod)
        explanation += "Prob Threshold Decay Ratio:{0}\n".format(GlobalConstants.PROBABILITY_THRESHOLD.decay)
    # explanation = "Tree. Gradient Type:{0} No threshold. Tree Degree:{1} " \
    #               "Initial Lr:{2} Decay Steps:{3} Decay Rate:{4} Total Param Count:{5} Wd:{6}".format(
    #                 GlobalConstants.GRADIENT_TYPE, GlobalConstants.TREE_DEGREE, GlobalConstants.INITIAL_LR,
    #                 GlobalConstants.DECAY_STEP, GlobalConstants.DECAY_RATE, total_param_count,
    #                 GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
    # explanation = "Wd, corrected. Baseline. C1:{0} C2:{1}: FC1:{2} " \
    #               "Initial Lr:{3} Decay Steps:{4} Decay Rate:{5} Total Param Count:{6} wd:{7}".format(
    #                GlobalConstants.NO_FILTERS_1, GlobalConstants.NO_FILTERS_2, GlobalConstants.NO_HIDDEN,
    #                GlobalConstants.INITIAL_LR, GlobalConstants.DECAY_STEP, GlobalConstants.DECAY_RATE,
    #                total_param_count, GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
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
    network = TreeNetwork(tree_degree=GlobalConstants.TREE_DEGREE,
                          node_build_funcs=[lenet3.root_func, lenet3.l1_func, lenet3.leaf_func],
                          grad_func=lenet3.grad_func,
                          create_new_variables=True)
    network.build_network()
    # dataset.reset()
    # Init
    init = tf.global_variables_initializer()
    # Grid search
    # wd_list = [0.0001 * x for n in range(0, 21) for x in itertools.repeat(n, 5)] # list(itertools.product(*list_of_lists))
    wd_list = [0.0]
    run_id = 0
    for wd in wd_list:
        print("********************NEW RUN:{0}********************".format(run_id))
        GlobalConstants.WEIGHT_DECAY_COEFFICIENT = wd
        experiment_id = DbLogger.get_run_id()
        explanation = get_explanation_string()
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
                if dataset.isNewEpoch:
                    print("Epoch Time={0}".format(total_time))
                    training_accuracy, training_confusion = \
                        network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.training,
                                                   run_id=experiment_id)
                    validation_accuracy, validation_confusion = \
                        network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.validation,
                                                   run_id=experiment_id)
                    DbLogger.write_into_table(rows=[(experiment_id, iteration_counter, epoch_id, training_accuracy,
                                                     validation_accuracy,
                                                     0.0, 0.0, "LeNet3")], table=DbLogger.logsTable, col_count=8)
                    DbLogger.write_into_table(rows=leaf_info_rows, table=DbLogger.leafInfoTable, col_count=4)
                    if GlobalConstants.SAVE_CONFUSION_MATRICES:
                        DbLogger.write_into_table(rows=training_confusion, table=DbLogger.confusionTable, col_count=6)
                        DbLogger.write_into_table(rows=validation_confusion, table=DbLogger.confusionTable, col_count=6)
                    leaf_info_rows = []
                    break
        test_accuracy, test_confusion = network.calculate_accuracy(sess=sess, dataset=dataset,
                                                                   dataset_type=DatasetTypes.test,
                                                                   run_id=experiment_id)
        DbLogger.write_into_table([(experiment_id, explanation, test_accuracy)], table=DbLogger.runResultsTable,
                                  col_count=3)
        if GlobalConstants.SAVE_CONFUSION_MATRICES:
            DbLogger.write_into_table(rows=test_confusion, table=DbLogger.confusionTable, col_count=6)
        print("X")
        run_id += 1


main()
