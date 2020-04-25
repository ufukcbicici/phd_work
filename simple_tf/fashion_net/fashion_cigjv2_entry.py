import os
import time
import tensorflow as tf
import numpy as np
from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from data_handling.fashion_mnist import FashionMnistDataSet
from simple_tf.cigj.jungle_gumbel_softmax import JungleGumbelSoftmax
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.fashion_net.fashion_net_cigj import FashionNetCigj
from simple_tf.fashion_net.fashion_net_cigj_v2 import FashionNetCigjV2
from simple_tf.global_params import GlobalConstants, AccuracyCalcType


def get_network(dataset, network_name):
    network = FashionNetCigjV2(
            node_build_funcs=[FashionNetCigjV2.build_lenet_node,
                              FashionNetCigjV2.build_lenet_node,
                              FashionNetCigjV2.build_lenet_node],
            h_dimensions=[2, 4],
            dataset=dataset,
            network_name="FashionNetCigjV2",
            level_params=GlobalConstants.CIGJ_V2_PARAMS)
    return network


def fashion_net_training():
    classification_wd = [0.0]
    decision_wd = [0.0]
    info_gain_balance_coeffs = [1.0] # [1.0, 2.0, 3.0, 4.0, 5.0]
    # classification_dropout_probs = sorted([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] *
    #                                       GlobalConstants.EXPERIMENT_MULTIPLICATION_FACTOR)
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
        network = get_network(dataset=dataset, network_name="GumbelSoftmaxFashionMNIST_CIGJ")
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
        series_id = 0
        # series_id = int(run_id / GlobalConstants.EXPERIMENT_MULTIPLICATION_FACTOR)
        explanation += "\n Series:{0}".format(series_id)
        DbLogger.write_into_table(rows=[(experiment_id, explanation)], table=DbLogger.runMetaData, col_count=2)
        sess.run(init)
        network.train(sess=sess, dataset=dataset, run_id=experiment_id)
        # try:
        #     network.train(sess=sess, dataset=dataset, run_id=experiment_id)
        # except Exception as e:
        #     DbLogger.write_into_table(rows=[(experiment_id, -1, "Error", str(e))], table=DbLogger.runKvStore,
        #                               col_count=4)
        tf.reset_default_graph()
        run_id += 1
