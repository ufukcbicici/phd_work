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
