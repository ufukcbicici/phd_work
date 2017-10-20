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


# tf.set_random_seed(1234)
# np_seed = 88
# np.random.seed(np_seed)


def main():
    # Build the network
    network = TreeNetwork(tree_degree=GlobalConstants.TREE_DEGREE,
                          node_build_funcs=[lenet3.root_func, lenet3.l1_func, lenet3.leaf_func],
                          create_new_variables=True,
                          data=GlobalConstants.TRAIN_DATA_TENSOR, label=GlobalConstants.TRAIN_LABEL_TENSOR)
    network.build_network(network_to_copy_from=None)
    # Do the training
    if GlobalConstants.USE_CPU:
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()
    dataset = MnistDataSet(validation_sample_count=10000, load_validation_from="validation_indices")
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
        total_param_count = 0
        for v in tf.trainable_variables():
            total_param_count += np.prod(v.get_shape().as_list())
        explanation = "Gradient Type:{0} No threshold. Tree Degree:{1} " \
                      "Initial Lr:{2} Decay Steps:{3} Decay Rate:{4} Total Param Count:{5} Wd:{6}".format(
                       GlobalConstants.GRADIENT_TYPE, GlobalConstants.TREE_DEGREE, GlobalConstants.INITIAL_LR,
                       GlobalConstants.DECAY_STEP, GlobalConstants.DECAY_RATE, total_param_count,
                       GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
        # explanation = "Wd, corrected. Baseline. C1:{0} C2:{1}: FC1:{2} " \
        #               "Initial Lr:{3} Decay Steps:{4} Decay Rate:{5} Total Param Count:{6} wd:{7}".format(
        #                GlobalConstants.NO_FILTERS_1, GlobalConstants.NO_FILTERS_2, GlobalConstants.NO_HIDDEN,
        #                GlobalConstants.INITIAL_LR, GlobalConstants.DECAY_STEP, GlobalConstants.DECAY_RATE,
        #                total_param_count, GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
        DbLogger.write_into_table(rows=[(experiment_id, explanation)], table=DbLogger.runMetaData,
                                  col_count=2)
        sess.run(init)
        # First loss
        network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.training, run_id=experiment_id)
        network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.validation,
                                   run_id=experiment_id)
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

# def experiment():
#     sess = tf.Session()
#     dataset = MnistDataSet(validation_sample_count=10000)
#     conv_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, NO_FILTERS_1], stddev=0.1, seed=SEED,
#                                                    dtype=DATA_TYPE), name="conv_weight")
#     conv_biases = tf.Variable(tf.constant(0.1, shape=[NO_FILTERS_1], dtype=DATA_TYPE), name="conv_bias")
#     hyperplane_weights = tf.Variable(tf.constant(value=0.1, shape=[IMAGE_SIZE * IMAGE_SIZE, TREE_DEGREE]),
#                                      name="hyperplane_weights")
#     hyperplane_biases = tf.Variable(tf.constant(0.1, shape=[TREE_DEGREE], dtype=DATA_TYPE), name="hyperplane_biases")
#     conv = tf.nn.conv2d(TRAIN_DATA_TENSOR, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
#     relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
#     pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#     flat_data = tf.contrib.layers.flatten(TRAIN_DATA_TENSOR)
#     activations = tf.matmul(flat_data, hyperplane_weights) + hyperplane_biases
#     arg_max_indices = tf.argmax(input=activations, axis=1)
#     maskTensorsDict = {}
#     outputsDict = {}
#     sumDict = {}
#     secondMasks = {}
#     secondMaskOutput = {}
#     activationsDict = {}
#     for index in range(TREE_DEGREE):
#         mask_vector = tf.equal(x=arg_max_indices, y=tf.constant(index, tf.int64), name="Mask_{0}".format(index))
#         maskTensorsDict[index] = mask_vector
#         outputsDict[index] = tf.boolean_mask(pool, mask_vector)
#         activationsDict[index] = tf.boolean_mask(activations, mask_vector)
#         flattened = tf.contrib.layers.flatten(outputsDict[index])
#         sum_vec = tf.reduce_sum(flattened, axis=1)
#         sumDict[index] = sum_vec
#         second_mask = tf.greater_equal(x=sum_vec, y=tf.constant(550.0, tf.float32))
#         secondMasks[index] = second_mask
#         secondMaskOutput[index] = tf.boolean_mask(activationsDict[index], second_mask)
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     samples, labels, indices_list = dataset.get_next_batch(batch_size=BATCH_SIZE)
#     samples = np.expand_dims(samples, axis=3)
#     for run_id in range(100):
#         feed_dict = {TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels}
#         results = sess.run([maskTensorsDict, outputsDict, activationsDict, arg_max_indices, sumDict, secondMaskOutput,
#                             secondMasks],
#                            feed_dict=feed_dict)
#         print("X")


main()
# experiment()

# conv1_weights = tf.Variable(
#             tf.truncated_normal([5, 5, GlobalConstants.NUM_CHANNELS, GlobalConstants.NO_FILTERS_1], stddev=0.1,
#                                 seed=GlobalConstants.SEED,
#                                 dtype=GlobalConstants.DATA_TYPE), name="conv1_weight")
# conv1_biases = tf.Variable(
#     tf.constant(0.1, shape=[GlobalConstants.NO_FILTERS_1], dtype=GlobalConstants.DATA_TYPE),
#     name="conv1_bias")
# conv2_weights = tf.Variable(
#     tf.truncated_normal([5, 5, GlobalConstants.NO_FILTERS_1, GlobalConstants.NO_FILTERS_2], stddev=0.1,
#                         seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
#     name="conv2_weight")
# conv2_biases = tf.Variable(
#     tf.constant(0.1, shape=[GlobalConstants.NO_FILTERS_2], dtype=GlobalConstants.DATA_TYPE),
#     name="conv2_bias")
# fc_weights_1 = tf.Variable(tf.truncated_normal(
#     [GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.IMAGE_SIZE // 4 * GlobalConstants.NO_FILTERS_2,
#      GlobalConstants.NO_HIDDEN],
#     stddev=0.1, seed=GlobalConstants.SEED, dtype=GlobalConstants.DATA_TYPE),
#                            name="fc_weights_1")
# fc_biases_1 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NO_HIDDEN], dtype=GlobalConstants.DATA_TYPE),
#                           name="fc_biases_1")
# fc_weights_2 = tf.Variable(
#     tf.truncated_normal([GlobalConstants.NO_HIDDEN, GlobalConstants.NUM_LABELS],
#                         stddev=0.1,
#                         seed=GlobalConstants.SEED,
#                         dtype=GlobalConstants.DATA_TYPE),
#     name="fc_weights_2")
# fc_biases_2 = tf.Variable(tf.constant(0.1, shape=[GlobalConstants.NUM_LABELS], dtype=GlobalConstants.DATA_TYPE),
#                           name="fc_biases_2")
# vars = tf.trainable_variables()
# total_param_count = 0
# for v in vars:
#     total_param_count += np.prod(v.get_shape().as_list())
# print("X")


# fc_weights_2 = tf.Variable(
#     tf.truncated_normal([GlobalConstants.NO_HIDDEN, GlobalConstants.NUM_LABELS],
#                         stddev=0.1,
#                         seed=GlobalConstants.SEED,
#                         dtype=GlobalConstants.DATA_TYPE),
#     name="fc_weights_2")
#
#
# config = tf.ConfigProto(device_count={'GPU': 0})
# sess = tf.Session(config=config)
# # Init
# init = tf.global_variables_initializer()
# sess.run(init)
#
# sliced = fc_weights_2[:, 2]
# res = sess.run([fc_weights_2, sliced])
# print("X")
