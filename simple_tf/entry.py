from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from algorithms.softmax_compresser import SoftmaxCompresser
from auxillary.constants import DatasetTypes

import time

import tensorflow as tf
import numpy as np

# MNIST
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DiscreteParameter, DecayingParameter
from data_handling.fashion_mnist import FashionMnistDataSet
from data_handling.mnist_data_set import MnistDataSet
from simple_tf import fashion_net_independent_h, lenet3, lenet_baseline
from simple_tf.global_params import GlobalConstants, AccuracyCalcType
from simple_tf.tree import TreeNetwork


# tf.set_random_seed(1234)
# np_seed = 88
# np.random.seed(np_seed)


def get_explanation_string(network):
    total_param_count = 0
    for v in network.variableManager.trainable_variables():
        total_param_count += np.prod(v.get_shape().as_list())

    # Tree
    explanation = "Fashion Mnist - Independent H - Tests - Parallel Dnns, Unbiased\n"
    # "(Lr=0.01, - Decay 1/(1 + i*0.0001) at each i. iteration)\n"
    explanation += "Batch Size:{0}\n".format(GlobalConstants.BATCH_SIZE)
    explanation += "Tree Degree:{0}\n".format(GlobalConstants.TREE_DEGREE_LIST)
    explanation += "Concat Trick:{0}\n".format(GlobalConstants.USE_CONCAT_TRICK)
    explanation += "Info Gain:{0}\n".format(GlobalConstants.USE_INFO_GAIN_DECISION)
    explanation += "Using Effective Sample Counts:{0}\n".format(GlobalConstants.USE_EFFECTIVE_SAMPLE_COUNTS)
    explanation += "Gradient Type:{0}\n".format(GlobalConstants.GRADIENT_TYPE)
    explanation += "Probability Threshold:{0}\n".format(GlobalConstants.USE_PROBABILITY_THRESHOLD)
    explanation += "********Lr Settings********\n"
    explanation += GlobalConstants.LEARNING_RATE_CALCULATOR.get_explanation()
    explanation += "********Lr Settings********\n"
    explanation += "Param Count:{0}\n".format(total_param_count)
    explanation += "Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
    explanation += "Decision Wd:{0}\n".format(GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT)
    explanation += "Residue Loss Coefficient:{0}\n".format(GlobalConstants.RESIDUE_LOSS_COEFFICIENT)
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
        explanation += "Noise Coefficient Initial Value:{0}\n".format(network.noiseCoefficientCalculator.value)
        explanation += "Noise Coefficient Decay Step:{0}\n".format(network.noiseCoefficientCalculator.decayPeriod)
        explanation += "Noise Coefficient Decay Ratio:{0}\n".format(network.noiseCoefficientCalculator.decay)
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
    explanation += "Classification Dropout Probability:{0}\n".format(GlobalConstants.CLASSIFICATION_DROPOUT_PROB)
    explanation += "Decision Dropout Probability:{0}\n".format(GlobalConstants.DECISION_DROPOUT_PROB)
    if GlobalConstants.USE_PROBABILITY_THRESHOLD:
        for node in network.topologicalSortedNodes:
            if node.isLeaf:
                continue
            explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
            explanation += node.probThresholdCalculator.get_explanation()
            explanation += "********Node{0} Probability Threshold Settings********\n".format(node.index)
    explanation += "Use Softmax Compression:{0}".format(GlobalConstants.USE_SOFTMAX_DISTILLATION)
    explanation += "F Conv1:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_FILTERS_1_SIZE,
                                                           GlobalConstants.FASHION_F_NUM_FILTERS_1)
    explanation += "F Conv2:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_FILTERS_2_SIZE,
                                                           GlobalConstants.FASHION_F_NUM_FILTERS_2)
    explanation += "F Conv3:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_FILTERS_3_SIZE,
                                                           GlobalConstants.FASHION_F_NUM_FILTERS_3)
    explanation += "F FC1:{0} Units\n".format(GlobalConstants.FASHION_F_FC_1)
    explanation += "F FC2:{0} Units\n".format(GlobalConstants.FASHION_F_FC_2)
    explanation += "F Residue FC:{0} Units\n".format(GlobalConstants.FASHION_F_RESIDUE)
    explanation += "H Conv1:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_H_FILTERS_1_SIZE,
                                                           GlobalConstants.FASHION_H_NUM_FILTERS_1)
    explanation += "H Conv2:{0}x{0}, {1} Filters\n".format(GlobalConstants.FASHION_H_FILTERS_2_SIZE,
                                                           GlobalConstants.FASHION_H_NUM_FILTERS_2)
    explanation += "FASHION_NO_H_FROM_F_UNITS_1:{0} Units\n".format(GlobalConstants.FASHION_NO_H_FROM_F_UNITS_1)
    explanation += "FASHION_NO_H_FROM_F_UNITS_2:{0} Units\n".format(GlobalConstants.FASHION_NO_H_FROM_F_UNITS_2)

    # Baseline
    # explanation = "Fashion Mnist Baseline. Double Dropout, Discrete learning rate\n"
    # explanation += "Batch Size:{0}\n".format(GlobalConstants.BATCH_SIZE)
    # explanation += "Gradient Type:{0}\n".format(GlobalConstants.GRADIENT_TYPE)
    # explanation += "Initial Lr:{0}\n".format(GlobalConstants.INITIAL_LR)
    # explanation += "Decay Steps:{0}\n".format(GlobalConstants.DECAY_STEP)
    # explanation += "Decay Rate:{0}\n".format(GlobalConstants.DECAY_RATE)
    # explanation += "Param Count:{0}\n".format(total_param_count)
    # explanation += "Model: {0}Conv - {1}Conv - {2}Conv - {3}FC - {4}FC\n".\
    #     format(GlobalConstants.FASHION_NUM_FILTERS_1, GlobalConstants.FASHION_NUM_FILTERS_2,
    #            GlobalConstants.FASHION_NUM_FILTERS_3, GlobalConstants.FASHION_FC_1, GlobalConstants.FASHION_FC_2)
    # explanation += "Conv1 Filters:{0} Conv2 Filters:{1} Conv3 Filters:{2}".\
    #     format(GlobalConstants.FASHION_FILTERS_1_SIZE, GlobalConstants.FASHION_FILTERS_2_SIZE,
    #            GlobalConstants.FASHION_FILTERS_3_SIZE)
    # explanation += "Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
    # explanation += "Dropout Prob:{0}\n".format(GlobalConstants.CLASSIFICATION_DROPOUT_PROB)
    return explanation


def main():
    # Do the training
    if GlobalConstants.USE_CPU:
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()
        dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
        # dataset = MnistDataSet(validation_sample_count=0, load_validation_from=None)
    # Build the network
    # network = TreeNetwork(tree_degree=GlobalConstants.TREE_DEGREE,
    #                       node_build_funcs=[lenet3.root_func, lenet3.l1_func, lenet3.leaf_func],
    #                       grad_func=lenet3.grad_func,
    #                       create_new_variables=True)

    # Mnist Baseline
    # network = TreeNetwork(  # tree_degree=GlobalConstants.TREE_DEGREE,
    #     node_build_funcs=[lenet_baseline.baseline],
    #     # node_build_funcs=[lenet_decision_connected_to_f.root_func, lenet_decision_connected_to_f.l1_func,
    #     #                   lenet_decision_connected_to_f.leaf_func],
    #     # node_build_funcs=[fashion_net_baseline.baseline],
    #     grad_func=lenet_baseline.grad_func,
    #     threshold_func=lenet_baseline.threshold_calculator_func,
    #     residue_func=lenet_baseline.residue_network_func,
    #     summary_func=lenet_baseline.tensorboard_func,
    #     degree_list=GlobalConstants.TREE_DEGREE_LIST)

    # Fashion Mnist H connected to F
    # network = TreeNetwork(
    #     node_build_funcs=[fashion_net_decision_connected_to_f.root_func,
    #                       fashion_net_decision_connected_to_f.l1_func,
    #                       fashion_net_decision_connected_to_f.leaf_func],
    #     grad_func=fashion_net_decision_connected_to_f.grad_func,
    #     threshold_func=fashion_net_decision_connected_to_f.threshold_calculator_func,
    #     residue_func=fashion_net_decision_connected_to_f.residue_network_func,
    #     summary_func=fashion_net_decision_connected_to_f.tensorboard_func,
    #     degree_list=GlobalConstants.TREE_DEGREE_LIST)

    # Fashion Mnist H independent
    network = TreeNetwork(
        node_build_funcs=[fashion_net_independent_h.root_func,
                          fashion_net_independent_h.l1_func,
                          fashion_net_independent_h.leaf_func],
        grad_func=fashion_net_independent_h.grad_func,
        threshold_func=fashion_net_independent_h.threshold_calculator_func,
        residue_func=fashion_net_independent_h.residue_network_func,
        summary_func=fashion_net_independent_h.tensorboard_func,
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
    classification_wd = [0.0]
    decision_wd = [0.0]
    info_gain_balance_coeffs = [5.0]
    classification_dropout_prob = [0.2]
    cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[classification_wd, decision_wd,
                                                                          info_gain_balance_coeffs,
                                                                          classification_dropout_prob])
    # del cartesian_product[0:10]
    # wd_list = [0.02]
    run_id = 0
    for tpl in cartesian_product:
        print("********************NEW RUN:{0}********************".format(run_id))
        # Restart the network; including all annealed parameters.
        GlobalConstants.WEIGHT_DECAY_COEFFICIENT = tpl[0]
        GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT = tpl[1]
        GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT = tpl[2]
        GlobalConstants.CLASSIFICATION_DROPOUT_PROB = 1.0 - tpl[3]
        # GlobalConstants.LEARNING_RATE_CALCULATOR = DecayingParameter(name="lr_calculator",
        #                                                              value=GlobalConstants.INITIAL_LR,
        #                                                              decay=GlobalConstants.DECAY_RATE,
        #                                                              decay_period=GlobalConstants.DECAY_STEP)
        GlobalConstants.LEARNING_RATE_CALCULATOR = DiscreteParameter(name="lr_calculator",
                                                                     value=GlobalConstants.INITIAL_LR,
                                                                     schedule=[(15000, 0.005),
                                                                               (30000, 0.0025),
                                                                               (40000, 0.00025)])
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
        network.reset_network(dataset=dataset, run_id=experiment_id)
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
                # if iteration_counter % 50 == 0:
                #     kv_rows = []
                #     for k, v in sample_counts.items():
                #         kv_rows.append((experiment_id, iteration_counter, k, np.asscalar(v)))
                #     DbLogger.write_into_table(rows=kv_rows, table=DbLogger.runKvStore, col_count=4)
                if dataset.isNewEpoch:
                    if (epoch_id + 1) % GlobalConstants.EPOCH_REPORT_PERIOD == 0:
                        print("Epoch Time={0}".format(total_time))
                        if not network.modeTracker.isCompressed:
                            training_accuracy, training_confusion = \
                                network.calculate_accuracy(sess=sess, dataset=dataset,
                                                           dataset_type=DatasetTypes.training,
                                                           run_id=experiment_id, iteration=iteration_counter,
                                                           calculation_type=AccuracyCalcType.regular)
                            validation_accuracy, validation_confusion = \
                                network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.test,
                                                           run_id=experiment_id, iteration=iteration_counter,
                                                           calculation_type=AccuracyCalcType.regular)
                            if not network.isBaseline:
                                validation_accuracy_corrected, validation_marginal_corrected = \
                                    network.calculate_accuracy(sess=sess, dataset=dataset,
                                                               dataset_type=DatasetTypes.test,
                                                               run_id=experiment_id,
                                                               iteration=iteration_counter,
                                                               calculation_type=
                                                               AccuracyCalcType.route_correction)
                            else:
                                validation_accuracy_corrected = 0.0
                                validation_marginal_corrected = 0.0
                        else:
                            training_accuracy, training_confusion = \
                                network.calculate_accuracy(sess=sess, dataset=dataset,
                                                           dataset_type=DatasetTypes.training,
                                                           run_id=experiment_id, iteration=iteration_counter,
                                                           calculation_type=AccuracyCalcType.regular)
                            validation_accuracy, validation_confusion = \
                                network.calculate_accuracy(sess=sess, dataset=dataset, dataset_type=DatasetTypes.test,
                                                           run_id=experiment_id, iteration=iteration_counter,
                                                           calculation_type=AccuracyCalcType.regular)
                            validation_accuracy_corrected = validation_accuracy
                        DbLogger.write_into_table(rows=[(experiment_id, iteration_counter, epoch_id, training_accuracy,
                                                         validation_accuracy, validation_accuracy_corrected,
                                                         0.0, 0.0, "XXX")], table=DbLogger.logsTable, col_count=9)
                        DbLogger.write_into_table(rows=leaf_info_rows, table=DbLogger.leafInfoTable, col_count=4)
                        if GlobalConstants.SAVE_CONFUSION_MATRICES and training_confusion is not None and \
                                        validation_confusion is not None:
                            DbLogger.write_into_table(rows=training_confusion, table=DbLogger.confusionTable,
                                                      col_count=7)
                            DbLogger.write_into_table(rows=validation_confusion, table=DbLogger.confusionTable,
                                                      col_count=7)
                        leaf_info_rows = []
                    break
            # Compress softmax classifiers
            if GlobalConstants.USE_SOFTMAX_DISTILLATION:
                do_compress = network.check_for_compression(dataset=dataset, run_id=experiment_id,
                                                            iteration=iteration_counter, epoch=epoch_id)
                if do_compress:
                    print("**********************Compressing the network**********************")
                    network.softmaxCompresser.compress_network_softmax(sess=sess)
                    print("**********************Compressing the network**********************")

        # test_accuracy, test_confusion = network.calculate_accuracy(sess=sess, dataset=dataset,
        #                                                            dataset_type=DatasetTypes.test,
        #                                                            run_id=experiment_id, iteration=iteration_counter,
        #                                                            calculation_type=AccuracyCalcType.regular)
        # result_rows = [(experiment_id, explanation, test_accuracy, "Regular")]
        # if not network.isBaseline:
        #     test_accuracy_corrected, test_marginal_corrected = \
        #         network.calculate_accuracy(sess=sess, dataset=dataset,
        #                                    dataset_type=DatasetTypes.test,
        #                                    run_id=experiment_id,
        #                                    iteration=iteration_counter,
        #                                    calculation_type=
        #                                    AccuracyCalcType.route_correction)
        #     # network.calculate_accuracy_with_residue_network(sess=sess, dataset=dataset, dataset_type=DatasetTypes.test)
        #     DbLogger.write_into_table(rows=[(experiment_id, iteration_counter,
        #                                      "Corrected Test Accuracy",
        #                                      test_accuracy_corrected)],
        #                               table=DbLogger.runKvStore, col_count=4)
        #     result_rows.append((experiment_id, explanation, test_accuracy_corrected, "Corrected"))
        #     result_rows.append((experiment_id, explanation, test_marginal_corrected, "Marginal"))
        # DbLogger.write_into_table(result_rows, table=DbLogger.runResultsTable, col_count=4)
        # if GlobalConstants.SAVE_CONFUSION_MATRICES and test_confusion is not None:
        #     DbLogger.write_into_table(rows=test_confusion, table=DbLogger.confusionTable, col_count=7)
        print("X")
        run_id += 1


main()
