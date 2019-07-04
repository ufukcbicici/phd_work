import tensorflow as tf
import numpy as np
import os
import time

from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import DiscreteParameter
from data_handling.fashion_mnist import FashionMnistDataSet
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.fashion_net import fashion_cign_connected_v2
from simple_tf.fashion_net.fashion_cign_connected_v2 import FashionCignV2
from simple_tf.global_params import GlobalConstants, AccuracyCalcType
from auxillary.constants import DatasetTypes


def get_explanation_string(network):
    total_param_count = 0
    for v in tf.trainable_variables():
        total_param_count += np.prod(v.get_shape().as_list())
    # Tree
    explanation = "Fashion Net - Multipath Optimization\n"
    # "(Lr=0.01, - Decay 1/(1 + i*0.0001) at each i. iteration)\n"
    explanation += "Using Fast Tree Version:{0}\n".format(GlobalConstants.USE_FAST_TREE_MODE)
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
    if not network.isBaseline:
        explanation += "********Decision Loss Weight Settings********\n"
        explanation += network.decisionLossCoefficientCalculator.get_explanation()
        explanation += "********Decision Loss Weight Settings********\n"
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
    explanation += "Classification Dropout Probability:{0}\n".format(GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB)
    explanation += "Decision Dropout Probability:{0}\n".format(network.decisionDropoutKeepProbCalculator.value)
    if GlobalConstants.USE_PROBABILITY_THRESHOLD:
        for node in network.topologicalSortedNodes:
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
    explanation += "Use Class Weighting:{0}\n".format(GlobalConstants.USE_CLASS_WEIGHTING)
    explanation += "Class Weight Running Average:{0}\n".format(GlobalConstants.CLASS_WEIGHT_RUNNING_AVERAGE)
    explanation += "Zero Label Count Epsilon:{0}\n".format(GlobalConstants.LABEL_EPSILON)
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
    # explanation += "Dropout Prob:{0}\n".format(GlobalConstants.CLASSIFICATION_DROPOUT_KEEP_PROB)
    return explanation


def fashion_net_training():
    dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
    dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
    classification_wd = [0.0]
    decision_wd = [0.0]
    info_gain_balance_coeffs = [5.0]
    # classification_dropout_probs = [0.15]
    classification_dropout_probs = [0.15]
    decision_dropout_probs = \
        [0.0]
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
        network = FashionCignV2(dataset=dataset, degree_list=GlobalConstants.TREE_DEGREE_LIST)
        network.build_network()
        init = tf.global_variables_initializer()
        print("********************NEW RUN:{0}********************".format(run_id))
        network.set_hyperparameters(weight_decay_coefficient=tpl[0],
                                    decision_weight_decay_coefficient=tpl[1],
                                    info_gain_balance_coefficient=tpl[2],
                                    classification_keep_probability=1.0 - tpl[3],
                                    decision_keep_probability=1.0 - tpl[4])
        experiment_id = DbLogger.get_run_id()
        explanation = get_explanation_string(network=network)
        series_id = int(run_id / GlobalConstants.EXPERIMENT_MULTIPLICATION_FACTOR)
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
                lr, sample_counts, is_open_indicators = network.update_params(sess=sess,
                                                                              dataset=dataset,
                                                                              epoch=epoch_id,
                                                                              iteration=iteration_counter)
                if all([lr, sample_counts, is_open_indicators]):
                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    print("Iteration:{0}".format(iteration_counter))
                    print("Lr:{0}".format(lr))
                    # Print sample counts (classification)
                    sample_count_str = "Classification:   "
                    for k, v in sample_counts.items():
                        sample_count_str += "[{0}={1}]".format(k, v)
                        node_index = network.get_node_from_variable_name(name=k).index
                        leaf_info_rows.append((node_index, np.asscalar(v), iteration_counter, experiment_id))
                    print(sample_count_str)
                    # Print node open indicators
                    indicator_str = ""
                    for k, v in is_open_indicators.items():
                        indicator_str += "[{0}={1}]".format(k, v)
                    print(indicator_str)
                    iteration_counter += 1
                if dataset.isNewEpoch:
                    # moving_results_1 = sess.run(moving_stat_vars)
                    if (epoch_id < GlobalConstants.TOTAL_EPOCH_COUNT - 10 and
                        (epoch_id + 1) % GlobalConstants.EPOCH_REPORT_PERIOD == 0) \
                            or epoch_id >= GlobalConstants.TOTAL_EPOCH_COUNT - 10:
                        print("Epoch Time={0}".format(total_time))
                        if not network.modeTracker.isCompressed:
                            training_accuracy, training_confusion = \
                                network.calculate_accuracy(sess=sess, dataset=dataset,
                                                           dataset_type=DatasetTypes.training,
                                                           run_id=experiment_id, iteration=iteration_counter,
                                                           calculation_type=AccuracyCalcType.regular)
                            validation_accuracy, validation_confusion = \
                                network.calculate_accuracy(sess=sess, dataset=dataset,
                                                           dataset_type=DatasetTypes.test,
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
                                if epoch_id >= GlobalConstants.TOTAL_EPOCH_COUNT - 10:
                                    network.calculate_accuracy(sess=sess, dataset=dataset,
                                                               dataset_type=DatasetTypes.test,
                                                               run_id=experiment_id,
                                                               iteration=iteration_counter,
                                                               calculation_type=
                                                               AccuracyCalcType.multi_path)
                            else:
                                validation_accuracy_corrected = 0.0
                                validation_marginal_corrected = 0.0
                            DbLogger.write_into_table(
                                rows=[(experiment_id, iteration_counter, epoch_id, training_accuracy,
                                       validation_accuracy, validation_accuracy_corrected,
                                       0.0, 0.0, "XXX")], table=DbLogger.logsTable, col_count=9)
                            # DbLogger.write_into_table(rows=leaf_info_rows, table=DbLogger.leafInfoTable, col_count=4)
                            if GlobalConstants.SAVE_CONFUSION_MATRICES:
                                DbLogger.write_into_table(rows=training_confusion, table=DbLogger.confusionTable,
                                                          col_count=7)
                                DbLogger.write_into_table(rows=validation_confusion, table=DbLogger.confusionTable,
                                                          col_count=7)
                        else:
                            training_accuracy_best_leaf, training_confusion_residue = \
                                network.calculate_accuracy(sess=sess, dataset=dataset,
                                                           dataset_type=DatasetTypes.training,
                                                           run_id=experiment_id, iteration=iteration_counter,
                                                           calculation_type=AccuracyCalcType.regular)
                            validation_accuracy_best_leaf, validation_confusion_residue = \
                                network.calculate_accuracy(sess=sess, dataset=dataset,
                                                           dataset_type=DatasetTypes.test,
                                                           run_id=experiment_id, iteration=iteration_counter,
                                                           calculation_type=AccuracyCalcType.regular)
                            DbLogger.write_into_table(rows=[(experiment_id, iteration_counter, epoch_id,
                                                             training_accuracy_best_leaf,
                                                             validation_accuracy_best_leaf,
                                                             validation_confusion_residue,
                                                             0.0, 0.0, "XXX")], table=DbLogger.logsTable,
                                                      col_count=9)
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
        # except Exception as e:
        #     print(e)
        #     print("ERROR!!!!")
        # Reset the computation graph
        tf.reset_default_graph()
        run_id += 1
