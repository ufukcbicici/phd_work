import tensorflow as tf
import numpy as np
import os
import time

from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from auxillary.parameters import FixedParameter, DiscreteParameter
from data_handling.cifar_dataset import CifarDataSet
from simple_tf.cifar_nets import cifar100_resnet_baseline, cign_resnet
from simple_tf.cign.cign_multi_gpu import CignMultiGpu
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.cign_with_sampling.cign_with_sampling import CignWithSampling
from simple_tf.global_params import GlobalConstants, AccuracyCalcType


def get_explanation_string(network):
    total_param_count = 0
    for v in tf.trainable_variables():
        total_param_count += np.prod(v.get_shape().as_list())

    # Tree
    explanation = "Resnet-50 Sampling CIGN Tests\n"
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
    explanation += "Classification Wd:{0}\n".format(GlobalConstants.WEIGHT_DECAY_COEFFICIENT)
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
    explanation += "Softmax Decay Initial:{0}\n".format(GlobalConstants.RESNET_SOFTMAX_DECAY_INITIAL)
    explanation += "Softmax Decay Coefficient:{0}\n".format(GlobalConstants.RESNET_SOFTMAX_DECAY_COEFFICIENT)
    explanation += "Softmax Decay Period:{0}\n".format(GlobalConstants.RESNET_SOFTMAX_DECAY_PERIOD)
    explanation += "Softmax Min Limit:{0}\n".format(GlobalConstants.RESNET_SOFTMAX_DECAY_MIN_LIMIT)
    explanation += "Softmax Test Temperature:{0}\n".format(GlobalConstants.RESNET_SOFTMAX_TEST_TEMPERATURE)
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
    explanation += "***** ResNet Parameters *****\n"
    explanation += str(GlobalConstants.RESNET_HYPERPARAMS)
    explanation += "Use Sampling CIGN:{0}".format(GlobalConstants.USE_SAMPLING_CIGN)
    return explanation


def get_network(dataset):
    if GlobalConstants.USE_SAMPLING_CIGN:
        GlobalConstants.USE_UNIFIED_BATCH_NORM = False
        print("USING SAMPLING CIGN!!!")
        network = CignWithSampling(
            node_build_funcs=[cign_resnet.root_func, cign_resnet.l1_func, cign_resnet.leaf_func],
            grad_func=cign_resnet.grad_func,
            threshold_func=cign_resnet.threshold_calculator_func_sampling,
            residue_func=cign_resnet.residue_network_func,
            summary_func=cign_resnet.tensorboard_func,
            degree_list=GlobalConstants.RESNET_TREE_DEGREES,
            dataset=dataset)
    else:
        network = FastTreeNetwork(
            node_build_funcs=[cign_resnet.root_func, cign_resnet.l1_func, cign_resnet.leaf_func],
            grad_func=cign_resnet.grad_func,
            threshold_func=cign_resnet.threshold_calculator_func,
            residue_func=cign_resnet.residue_network_func,
            summary_func=cign_resnet.tensorboard_func,
            degree_list=GlobalConstants.RESNET_TREE_DEGREES,
            dataset=dataset)
    return network


def cifar100_training():
    # classification_wd = [0.00005 * i for i in range(21)] * 3
    # classification_wd = sorted(classification_wd)
    classification_wd = [0.00015] * 4
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
        dataset = CifarDataSet(session=sess,
                               validation_sample_count=0, load_validation_from=None)
        dataset.set_curr_session(sess=sess)
        # dataset = CifarDataSet(validation_sample_count=0, load_validation_from=None)
        dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
        network = get_network(dataset=dataset)
        GlobalConstants.LEARNING_RATE_CALCULATOR = DiscreteParameter(name="lr_calculator",
                                                                     value=GlobalConstants.INITIAL_LR,
                                                                     schedule=[(40000,  0.01),
                                                                               (70000,  0.001),
                                                                               (100000, 0.0001)])
        network.build_network()
        # Init
        init = tf.global_variables_initializer()
        print("********************NEW RUN:{0}********************".format(run_id))
        # Restart the network; including all annealed parameters.
        GlobalConstants.WEIGHT_DECAY_COEFFICIENT = tpl[0]
        GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT = tpl[1]
        GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT = tpl[2]
        GlobalConstants.CLASSIFICATION_DROPOUT_PROB = 1.0 - tpl[3]
        network.decisionDropoutKeepProbCalculator = FixedParameter(name="decision_dropout_prob", value=1.0 - tpl[4])
        network.learningRateCalculator = GlobalConstants.LEARNING_RATE_CALCULATOR
        network.thresholdFunc(network=network)
        experiment_id = DbLogger.get_run_id()
        explanation = get_explanation_string(network=network)
        series_id = int(run_id / 4)
        explanation += "\n Series:{0}".format(series_id)
        DbLogger.write_into_table(rows=[(experiment_id, explanation)], table=DbLogger.runMetaData, col_count=2)
        sess.run(init)
        network.reset_network(dataset=dataset, run_id=experiment_id)
        # moving_stat_vars = [var for var in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES) if "moving" in var.name]
        # moving_results_0 = sess.run(moving_stat_vars)
        iteration_counter = 0
        for epoch_id in range(GlobalConstants.TOTAL_EPOCH_COUNT):
            # An epoch is a complete pass on the whole dataset.
            dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
            print("*************Epoch {0}*************".format(epoch_id))
            total_time = 0.0
            leaf_info_rows = []
            while True:
                start_time = time.time()
                lr, sample_counts, is_open_indicators = network.update_params_with_momentum(sess=sess,
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
                    if (epoch_id < GlobalConstants.TOTAL_EPOCH_COUNT-30 and
                            (epoch_id + 1) % GlobalConstants.EPOCH_REPORT_PERIOD == 0) \
                            or epoch_id >= GlobalConstants.TOTAL_EPOCH_COUNT-30:
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

    # dataset.visualize_sample(sample_index=150)
    print("X")


def cifar100_multi_gpu_training():
    classification_wd = [0.0]
    decision_wd = [0.0]
    info_gain_balance_coeffs = [1.0]
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
        dataset = CifarDataSet(session=sess,
                               validation_sample_count=0, load_validation_from=None)
        dataset.set_curr_session(sess=sess)
        # dataset = CifarDataSet(validation_sample_count=0, load_validation_from=None)
        dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
        # network = get_network(dataset=dataset)
        network = CignMultiGpu(
            node_build_funcs=[cign_resnet.root_func, cign_resnet.l1_func, cign_resnet.leaf_func],
            grad_func=cign_resnet.grad_func,
            threshold_func=cign_resnet.threshold_calculator_func,
            residue_func=cign_resnet.residue_network_func,
            summary_func=cign_resnet.tensorboard_func,
            degree_list=GlobalConstants.RESNET_TREE_DEGREES,
            dataset=dataset)
        GlobalConstants.LEARNING_RATE_CALCULATOR = DiscreteParameter(name="lr_calculator",
                                                                     value=GlobalConstants.INITIAL_LR,
                                                                     schedule=[(40000,  0.01),
                                                                               (70000,  0.001),
                                                                               (100000, 0.0001)])
        network.build_network()
        print("X")



# main()
# main_fast_tree()
# ensemble_training()
# cifar100_training()
# xxx
