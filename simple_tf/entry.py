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
from data_handling.mnist_data_set import MnistDataSet
from simple_tf.global_params import GlobalConstants
from simple_tf.tree import TreeNetwork
import simple_tf.lenet3 as lenet3


tf.set_random_seed(1234)
np_seed = 88
np.random.seed(np_seed)


def main():
    # Build the network
    # network = TreeNetwork(tree_degree=2, node_build_funcs=[baseline], create_new_variables=True,
    #                                data=TRAIN_DATA_TENSOR, label=TRAIN_LABEL_TENSOR)
    # network.build_network(network_to_copy_from=None)
    network = TreeNetwork(tree_degree=GlobalConstants.TREE_DEGREE, node_build_funcs=[lenet3.root_func, lenet3.l1_func, lenet3.leaf_func],
                          create_new_variables=True,
                          data=GlobalConstants.TRAIN_DATA_TENSOR, label=GlobalConstants.TRAIN_LABEL_TENSOR)
    network.build_network(network_to_copy_from=None)
    # test_network = TreeNetwork(tree_degree=2, node_build_funcs=[baseline], create_new_variables=False,
    #                            data=TEST_DATA_TENSOR, label=TEST_LABEL_TENSOR)
    # test_network.build_network(network_to_copy_from=training_network)
    # Do the training
    if GlobalConstants.USE_CPU:
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()
    dataset = MnistDataSet(validation_sample_count=10000)
    # Acquire the losses for training
    # loss_list = []
    # vars = tf.trainable_variables()
    # var_dict = {v.name:  v for v in vars}
    # var_names = [v.name for v in vars]
    # Train
    # Setting the optimizer
    # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(network.finalLoss, global_step=global_counter)
    # Init variables
    init = tf.global_variables_initializer()
    sess.run(init)
    # Experiment
    # samples, labels, indices_list = dataset.get_next_batch(batch_size=BATCH_SIZE)
    # samples = np.expand_dims(samples, axis=3)
    # eval
    # params = {v.name: v for node in network.topologicalSortedNodes for v in node.variablesList}
    # param_values = sess.run(params, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    # node3_eval = {k: v for k, v in network.evalDict.items() if "Node3" in k}
    # node4_eval = {k: v for k, v in network.evalDict.items() if "Node4" in k}
    # node5_eval = {k: v for k, v in network.evalDict.items() if "Node5" in k}
    # node6_eval = {k: v for k, v in network.evalDict.items() if "Node6" in k}

    # for run_id in range(100):
    #     # results = sess.run(network.evalDict, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    #     results3 = sess.run(node3_eval, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    #     print("Completed: Node3")
    # for run_id in range(100):
    #     results4 = sess.run(node4_eval, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    #     print("Completed: Node4")
    # for run_id in range(100):
    #     results5 = sess.run(node5_eval, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    #     print("Completed: Node5")
    # for run_id in range(100):
    #     results6 = sess.run(node6_eval, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    #     print("Completed: Node6")
    #
    # for run_id in range(100):
    #     results6 = sess.run(node6_eval, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    #     print("Completed: Node6")
    # for run_id in range(100):
    #     start_time = time.time()
    #     results = sess.run(network.evalDict, feed_dict={TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels})
    #     elapsed_time = time.time() - start_time
    #     print("elapsed time={0}".format(elapsed_time))
    #     for k, v in results.items():
    #         if "sample_count" in k:
    #             print("{0}={1}".format(k, v))
    #     print("X")

    # First loss
    network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.training)
    network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.validation)
    lr = 9000.0
    iteration_counter = 0
    for epoch_id in range(GlobalConstants.EPOCH_COUNT):
        # An epoch is a complete pass on the whole dataset.
        dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
        print("*************Epoch {0}*************".format(epoch_id))
        total_time = 0.0
        while True:
            # samples, labels, indices_list = dataset.get_next_batch(batch_size=BATCH_SIZE)
            # samples = np.expand_dims(samples, axis=3)
            # start_time = time.time()
            # feed_dict = {TRAIN_DATA_TENSOR: samples, TRAIN_LABEL_TENSOR: labels}
            # results = sess.run([gradients, sample_count_tensors], feed_dict=feed_dict)
            start_time = time.time()
            sample_counts, lr = network.update_params_with_momentum(sess=sess, dataset=dataset,
                                                                    iteration=iteration_counter)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            print("Iteration:{0}".format(iteration_counter))
            print("Lr:{0}".format(lr))
            # Print sample counts
            for k, v in sample_counts.items():
                print("{0}={1}".format(k, v))
            # print("Iteration:{0}".format(results[1]))
            # if abs(results[0] - lr) > 1e-10:
            #     print("Learning rate changed to {0}".format(results[0]))
            #     lr = results[0]
            # print("lr={0}".format(results[0]))
            # print("global counter={0}".format(results[1]))
            iteration_counter += 1
            if dataset.isNewEpoch:
                print("Epoch Time={0}".format(total_time))
                network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.training)
                network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.validation)
                break
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
