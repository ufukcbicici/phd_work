import os
import time
import tensorflow as tf
import numpy as np
from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import FixedParameter
from data_handling.fashion_mnist import FashionMnistDataSet
from simple_tf.cigj.jungle import Jungle
from simple_tf.fashion_net.fashion_net_cigj import FashionNetCigj
from simple_tf.global_params import GlobalConstants


def get_explanation_string(network):
    total_param_count = 0
    for v in tf.trainable_variables():
        total_param_count += np.prod(v.get_shape().as_list())
    # Tree
    explanation = "CIGJ Fashion MNIST Tests\n"
    # "(Lr=0.01, - Decay 1/(1 + i*0.0001) at each i. iteration)\n"
    explanation += "Batch Size:{0}\n".format(GlobalConstants.BATCH_SIZE)
    explanation += "Jungle Degree Degree:{0}\n".format(GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST)
    explanation += "********Lr Settings********\n"
    explanation += GlobalConstants.LEARNING_RATE_CALCULATOR.get_explanation()
    explanation += "********Lr Settings********\n"
    if not network.isBaseline:
        explanation += "********Decision Loss Weight Settings********\n"
        explanation += network.decisionLossCoefficientCalculator.get_explanation()
        explanation += "********Decision Loss Weight Settings********\n"
    explanation += "Batch Norm Decay:{0}\n".format(GlobalConstants.BATCH_NORM_DECAY)
    explanation += "Param Count:{0}\n".format(total_param_count)
    explanation += "Classification Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
    explanation += "Decision Wd:{0}\n".format(GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT)
    explanation += "Info Gain Loss Lambda:{0}\n".format(GlobalConstants.DECISION_LOSS_COEFFICIENT)
    explanation += "Use Batch Norm Before Decisions:{0}\n".format(GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING)
    explanation += "Softmax Decay Initial:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_INITIAL)
    explanation += "Softmax Decay Coefficient:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_COEFFICIENT)
    explanation += "Softmax Decay Period:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_PERIOD)
    explanation += "Softmax Min Limit:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_MIN_LIMIT)
    explanation += "Softmax Test Temperature:{0}\n".format(GlobalConstants.SOFTMAX_TEST_TEMPERATURE)
    explanation += "Info Gain Balance Coefficient:{0}\n".format(GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT)
    explanation += "Classification Dropout Probability:{0}\n".format(GlobalConstants.CLASSIFICATION_DROPOUT_PROB)
    explanation += "Decision Dropout Probability:{0}\n".format(network.decisionDropoutKeepProbCalculator.value)
    # if GlobalConstants.USE_PROBABILITY_THRESHOLD:
    #     for node in network.topologicalSortedNodes:
    #         if node.isLeaf:
    #             continue
    #         explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
    #         explanation += node.probThresholdCalculator.get_explanation()
    #         explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
    return explanation


def cigj_training():
    classification_wd = [0.0]
    decision_wd = [0.0]
    info_gain_balance_coeffs = [1.0]
    # classification_dropout_probs = [0.15]
    classification_dropout_probs = [0.0]
    decision_dropout_probs = [0.0]
    cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[classification_wd,
                                                                          decision_wd,
                                                                          info_gain_balance_coeffs,
                                                                          classification_dropout_probs,
                                                                          decision_dropout_probs])
    run_id = 0
    for tpl in cartesian_product:
        # try:
        # Session initialization
        if GlobalConstants.USE_CPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            config = tf.ConfigProto(device_count={'GPU': 0})
            sess = tf.Session(config=config)
        else:
            sess = tf.Session()
        dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
        dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
        jungle = Jungle(
            node_build_funcs=[FashionNetCigj.f_conv_layer_func,
                              FashionNetCigj.f_conv_layer_func,
                              FashionNetCigj.f_conv_layer_func,
                              FashionNetCigj.f_fc_layer_func,
                              FashionNetCigj.f_leaf_func],
            h_funcs=[FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func],
            grad_func=None,
            threshold_func=FashionNetCigj.threshold_calculator_func,
            residue_func=None, summary_func=None,
            degree_list=GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST, dataset=dataset)
        sess = jungle.get_session()
        init = tf.global_variables_initializer()
        GlobalConstants.WEIGHT_DECAY_COEFFICIENT = tpl[0]
        GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT = tpl[1]
        GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT = tpl[2]
        GlobalConstants.CLASSIFICATION_DROPOUT_PROB = 1.0 - tpl[3]
        jungle.decisionDropoutKeepProbCalculator = FixedParameter(name="decision_dropout_prob", value=1.0 - tpl[4])
        jungle.learningRateCalculator = GlobalConstants.LEARNING_RATE_CALCULATOR
        jungle.thresholdFunc(network=jungle)
        experiment_id = DbLogger.get_run_id()
        explanation = get_explanation_string(network=jungle)
        series_id = int(run_id / 6)
        explanation += "\n Series:{0}".format(series_id)
        DbLogger.write_into_table(rows=[(experiment_id, explanation)], table=DbLogger.runMetaData, col_count=2)
        sess.run(init)
        iteration_counter = 0
        for epoch_id in range(GlobalConstants.TOTAL_EPOCH_COUNT):
            # An epoch is a complete pass on the whole dataset.
            dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
            print("*************Epoch {0}*************".format(epoch_id))
            total_time = 0.0
            leaf_info_rows = []
            while True:
                start_time = time.time()
                lr, sample_counts, is_open_indicators = jungle.update_params_with_momentum(sess=sess,
                                                                                            dataset=dataset,
                                                                                            epoch=epoch_id,
                                                                                            iteration=iteration_counter)



















    classification_wd = [0.0]
    decision_wd = [0.0]
    info_gain_balance_coeffs = [1.0]
    classification_dropout_probs = [0.0]
    decision_dropout_probs = [0.0]














    # init = tf.global_variables_initializer()
    # sess.run(init)
    # results, minibatch = jungle.eval_network(sess=sess, dataset=dataset, use_masking=True)
    # # Check consistency of partitioning - stitching
    # for node in jungle.nodes.values():
    #     if node.nodeType == NodeType.h_node:
    #         if UtilityFuncs.get_variable_name(name="stitchedIndices", node=node) in results:
    #             stitched_indices = results[UtilityFuncs.get_variable_name(name="stitchedIndices", node=node)]
    #             assert np.array_equal(np.arange(GlobalConstants.CURR_BATCH_SIZE), stitched_indices)
    #         if UtilityFuncs.get_variable_name(name="stitchedLabels", node=node) in results:
    #             stitched_labels = results[UtilityFuncs.get_variable_name(name="stitchedLabels", node=node)]
    #             assert np.array_equal(minibatch.labels, stitched_labels)
    # print("X")

    # histogram = np.zeros(shape=(GlobalConstants.EVAL_BATCH_SIZE, 3))
    # minibatch = dataset.get_next_batch(batch_size=GlobalConstants.EVAL_BATCH_SIZE)
    # probs = None
    # for i in range(10000):
    #     if i % 100 == 0:
    #         print(i)
    #     # results, _ = jungle.eval_network(sess=sess, dataset=dataset, use_masking=True)
    #     results, _ = jungle.eval_minibatch(sess=sess, minibatch=minibatch, use_masking=True)
    #     selected_indices = results["Node1_indices_tensor"]
    #     if probs is None:
    #         probs = results["Node1_p(n|x)"]
    #     else:
    #         assert np.array_equal(probs, results["Node1_p(n|x)"])
    #     histogram[np.arange(histogram.shape[0]), selected_indices] += 1
    # sampled_probs = histogram / np.reshape(np.sum(histogram, axis=1), newshape=(histogram.shape[0], 1))
    # print("X")

    # jungle.print_trellis_structure()


cigj_training()

# with tf.control_dependencies([shape_assign_op]):
#     set_batch_size_op = tf.assign(shape_tensor[0], batch_size_tensor)
#     with tf.control_dependencies([set_batch_size_op]):
#         x = tf.identity(shape_tensor)
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# result = sess.run([x], feed_dict={sparse_tensor: sparse_arr, batch_size_tensor: batch_size})
# print("X")
# square_sparse = tf.square(sparse_tensor)
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# with tf.control_dependencies(shape_assign_op):
#     # with tf.control_dependencies(set_batch_size_op):
#     results = sess.run([square_sparse], feed_dict={sparse_tensor: sparse_arr, indices_tensor: indices,
#                                                    batch_size_tensor: batch_size})
#     print("X")

# zero_index = tf.constant(0)


# tf.scatter_update()
# shape_assign_op = tf.assign(shape_tensor, tf.shape(sparse_tensor))
# set_batch_size_op = tf.assign(shape_tensor[0], batch_size_tensor)
# with tf.control_dependencies(shape_assign_op):
# with tf.control_dependencies(set_batch_size_op):
# square = tf.square(sparse_arr)


# batch_size = tf.placeholder(dtype=tf.int32)
# prob_tensor = tf.placeholder(dtype=tf.float32)
# prob_arr = np.array([[0.3, 0.4, 0.3], [0.7, 0.1, 0.2], [0.25, 0.25, 0.5], [0.95, 0.05, 0.05], [0.1, 0.2, 0.7]])
# dist = tf.distributions.Categorical(probs=prob_tensor)
# samples = dist.sample()
# one_hot_samples = tf.one_hot(indices=samples, depth=3, axis=-1)
# sess = tf.Session()
# samples_arr = None
# for i in range(100000):
#     print(i)
#     res = sess.run([samples, one_hot_samples], feed_dict={batch_size: 100000, prob_tensor: prob_arr})
#     if i == 0:
#         samples_arr = res[0]
#         samples_arr = np.expand_dims(samples_arr, axis=1)
#     else:
#         curr_samples = res[0]
#         curr_samples = np.expand_dims(curr_samples, axis=1)
#         samples_arr = np.concatenate((samples_arr, curr_samples), axis=1)
# print("X")
