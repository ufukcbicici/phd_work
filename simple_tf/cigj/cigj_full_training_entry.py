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
from simple_tf.cigj.jungle_gumbel_softmax import JungleGumbelSoftmax
from simple_tf.cigj.jungle_no_stitch import JungleNoStitch
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.fashion_net.fashion_net_cigj import FashionNetCigj
from simple_tf.global_params import GlobalConstants, AccuracyCalcType


def get_explanation_string(network):
    total_param_count = 0
    for v in tf.trainable_variables():
        total_param_count += np.prod(v.get_shape().as_list())
    # Tree
    explanation = "CIGJ Fashion MNIST Gumbel-Softmax Tests: 128 Sized H Features\n"
    explanation += "Network Type:{0}\n".format(network.__class__)
    # "(Lr=0.01, - Decay 1/(1 + i*0.0001) at each i. iteration)\n"
    explanation += "Batch Size:{0}\n".format(GlobalConstants.BATCH_SIZE)
    explanation += "Jungle Degree Degree:{0}\n".format(GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST)
    explanation += "Optimizer:{0}\n".format(GlobalConstants.OPTIMIZER_TYPE)
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
    # explanation += "Softmax Decay Initial:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_INITIAL)
    # explanation += "Softmax Decay Coefficient:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_COEFFICIENT)
    # explanation += "Softmax Decay Period:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_PERIOD)
    # explanation += "Softmax Min Limit:{0}\n".format(GlobalConstants.SOFTMAX_DECAY_MIN_LIMIT)
    # explanation += "Softmax Test Temperature:{0}\n".format(GlobalConstants.SOFTMAX_TEST_TEMPERATURE)

    explanation += "********Softmax Decay Settings********\n"
    for node in network.topologicalSortedNodes:
        if node.nodeType == NodeType.h_node:
            explanation += "********Node{0} Softmax Decay********\n".format(node.index)
            explanation += node.softmaxDecayCalculator.get_explanation()
            explanation += "********Node{0} Softmax Decay********\n".format(node.index)
    explanation += "********Softmax Decay Settings********\n"

    explanation += "********Gumbel Softmax Temperature Settings********\n"
    for node in network.topologicalSortedNodes:
        if node.nodeType == NodeType.h_node:
            explanation += "********Node{0} Gumbel Softmax Temperature********\n".format(node.index)
            explanation += node.gumbelSoftmaxTemperatureCalculator.get_explanation()
            explanation += "********Node{0} Gumbel Softmax Temperature********\n".format(node.index)
    explanation += "********Gumbel Softmax Temperature Settings********\n"

    explanation += "Info Gain Balance Coefficient:{0}\n".format(GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT)
    explanation += "Classification Dropout Probability:{0}\n".format(GlobalConstants.CLASSIFICATION_DROPOUT_PROB)
    explanation += "Decision Dropout Probability:{0}\n".format(GlobalConstants.DECISION_DROPOUT_PROB)
    explanation += "H Feature Sizes:{0}\n".format(GlobalConstants.CIGJ_FASHION_NET_H_FEATURES)
    explanation += "H Pooling Sizes:{0}\n".format(GlobalConstants.CIGJ_FASHION_NET_H_POOL_SIZES)
    # explanation += "Decision Dropout Probability:{0}\n".format(network.decisionDropoutKeepProbCalculator.value)
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
    info_gain_balance_coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    # # classification_dropout_probs = [0.15]
    classification_dropout_probs = sorted([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] *
                                          GlobalConstants.EXPERIMENT_MULTIPLICATION_FACTOR)
    # info_gain_balance_coeffs = [1.0]
    # classification_dropout_probs = [0.15]
    # classification_dropout_probs = sorted([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    #                                       * GlobalConstants.EXPERIMENT_MULTIPLICATION_FACTOR)
    # classification_dropout_probs = [0.0]
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
        jungle = JungleGumbelSoftmax(
            node_build_funcs=[FashionNetCigj.f_conv_layer_func,
                              FashionNetCigj.f_conv_layer_func,
                              FashionNetCigj.f_conv_layer_func,
                              FashionNetCigj.f_fc_layer_func,
                              FashionNetCigj.f_leaf_func],
            h_funcs=[FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func],
            grad_func=None,
            threshold_func=FashionNetCigj.threshold_calculator_gumbel_softmax_func,
            residue_func=None, summary_func=None,
            degree_list=GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST, dataset=dataset)
        sess = jungle.get_session()
        init = tf.global_variables_initializer()
        GlobalConstants.WEIGHT_DECAY_COEFFICIENT = tpl[0]
        GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT = tpl[1]
        GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT = tpl[2]
        GlobalConstants.CLASSIFICATION_DROPOUT_PROB = 1.0 - tpl[3]
        GlobalConstants.DECISION_DROPOUT_PROB = 1.0 - tpl[4]
        # jungle.decisionDropoutKeepProbCalculator = FixedParameter(name="decision_dropout_prob", value=1.0 - tpl[4])
        jungle.learningRateCalculator = GlobalConstants.LEARNING_RATE_CALCULATOR
        jungle.thresholdFunc(network=jungle)
        experiment_id = DbLogger.get_run_id()
        explanation = get_explanation_string(network=jungle)
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
                lr, sample_counts, is_open_indicators = jungle.update_params_with_momentum(sess=sess,
                                                                                           dataset=dataset,
                                                                                           epoch=epoch_id,
                                                                                           iteration=iteration_counter)
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                print("Iteration:{0}".format(iteration_counter))
                print("Lr:{0}".format(lr))
                # Print sample counts (classification)
                sample_count_str = "Classification:   "
                for k, v in sample_counts.items():
                    if np.isnan(v):
                        print("NAN!!!")
                    sample_count_str += "[{0}={1}]".format(k, v)
                    node_index = jungle.get_node_from_variable_name(name=k).index
                    leaf_info_rows.append((node_index, np.asscalar(v), iteration_counter, experiment_id))
                print(sample_count_str)
                # Print node open indicators
                indicator_str = ""
                for k, v in is_open_indicators.items():
                    indicator_str += "[{0}={1}]".format(k, v)
                print(indicator_str)
                iteration_counter += 1
                if dataset.isNewEpoch:
                    if (epoch_id < GlobalConstants.TOTAL_EPOCH_COUNT - 15 and
                        (epoch_id + 1) % GlobalConstants.EPOCH_REPORT_PERIOD == 0) \
                            or epoch_id >= GlobalConstants.TOTAL_EPOCH_COUNT - 15:
                        training_accuracy, training_confusion = \
                            jungle.calculate_accuracy(sess=sess, dataset=dataset,
                                                      dataset_type=DatasetTypes.training,
                                                      run_id=experiment_id, iteration=iteration_counter,
                                                      calculation_type=AccuracyCalcType.regular)
                        validation_accuracy, validation_confusion = \
                            jungle.calculate_accuracy(sess=sess, dataset=dataset,
                                                      dataset_type=DatasetTypes.test,
                                                      run_id=experiment_id, iteration=iteration_counter,
                                                      calculation_type=AccuracyCalcType.regular)
                        DbLogger.write_into_table(
                            rows=[(experiment_id, iteration_counter,
                                   epoch_id,
                                   training_accuracy,
                                   validation_accuracy, validation_accuracy,
                                   0.0, 0.0, "XXX")], table=DbLogger.logsTable, col_count=9)
                    break
        tf.reset_default_graph()
        run_id += 1


# cigj_training()
