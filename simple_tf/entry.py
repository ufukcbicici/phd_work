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

# MNIST
from auxillary.db_logger import DbLogger
from data_handling.mnist_data_set import MnistDataSet
from simple_tf.global_params import GlobalConstants
from simple_tf.tree import TreeNetwork
import simple_tf.lenet3 as lenet3


# tf.set_random_seed(1234)
# np_seed = 88
# np.random.seed(np_seed)


def main():
    # Build the network
    network = TreeNetwork(tree_degree=GlobalConstants.TREE_DEGREE, node_build_funcs=[lenet3.root_func, lenet3.l1_func, lenet3.leaf_func],
                          create_new_variables=True,
                          data=GlobalConstants.TRAIN_DATA_TENSOR, label=GlobalConstants.TRAIN_LABEL_TENSOR)
    network.build_network(network_to_copy_from=None)
    # Do the training
    if GlobalConstants.USE_CPU:
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()
    dataset = MnistDataSet(validation_sample_count=10000)
    # Init
    init = tf.global_variables_initializer()
    for run_id in range(100):
        print("********************NEW RUN:{0}********************".format(run_id))
        experiment_id = DbLogger.get_run_id()
        explanation = "Gradient Type:{0} No threshold.".format(GlobalConstants.GRADIENT_TYPE)
        DbLogger.write_into_table(rows=[(experiment_id, explanation)], table=DbLogger.runMetaData,
                                  col_count=2)
        sess.run(init)
        # First loss
        network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.training)
        network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.validation)
        iteration_counter = 0
        for epoch_id in range(GlobalConstants.EPOCH_COUNT):
            # An epoch is a complete pass on the whole dataset.
            dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
            print("*************Epoch {0}*************".format(epoch_id))
            total_time = 0.0
            while True:
                start_time = time.time()
                sample_counts, lr, is_open_indicators = network.update_params_with_momentum(sess=sess, dataset=dataset,
                                                                        iteration=iteration_counter)
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                print("Iteration:{0}".format(iteration_counter))
                print("Lr:{0}".format(lr))
                # Print sample counts
                for k, v in sample_counts.items():
                    print("{0}={1}".format(k, v))
                for k, v in is_open_indicators.items():
                    print("{0}={1}".format(k, v))
                iteration_counter += 1
                if dataset.isNewEpoch:
                    print("Epoch Time={0}".format(total_time))
                    training_accuracy = \
                        network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.training)
                    validation_accuracy = \
                        network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.validation)
                    DbLogger.write_into_table(rows=[(experiment_id, iteration_counter, epoch_id, training_accuracy,
                                                     validation_accuracy,
                                                     0.0, 0.0, "LeNet3")], table=DbLogger.logsTable, col_count=8)
                    break
        test_accuracy = network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.test)
        DbLogger.write_into_table([(experiment_id, explanation, test_accuracy)], table=DbLogger.runResultsTable,
                                  col_count=3)
        print("X")


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
