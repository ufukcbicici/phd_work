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
        network.set_training_parameters()
        network.build_network()
        init = tf.global_variables_initializer()
        print("********************NEW RUN:{0}********************".format(run_id))
        network.set_hyperparameters(weight_decay_coefficient=tpl[0],
                                    decision_weight_decay_coefficient=tpl[1],
                                    info_gain_balance_coefficient=tpl[2],
                                    classification_keep_probability=1.0 - tpl[3],
                                    decision_keep_probability=1.0 - tpl[4])
        experiment_id = DbLogger.get_run_id()
        explanation = network.get_explanation_string()
        series_id = int(run_id / GlobalConstants.EXPERIMENT_MULTIPLICATION_FACTOR)
        explanation += "\n Series:{0}".format(series_id)
        DbLogger.write_into_table(rows=[(experiment_id, explanation)], table=DbLogger.runMetaData, col_count=2)
        sess.run(init)
        network.train(sess=sess, dataset=dataset, run_id=experiment_id)
        tf.reset_default_graph()
        run_id += 1
