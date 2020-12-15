import tensorflow as tf
import os

from algorithms.nnrf_computation_statistics_calculator import NNRFComputationStatisticsCalculator
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from data_handling.fashion_mnist import FashionMnistDataSet
from data_handling.mnist_data_set import MnistDataSet
from data_handling.usps_dataset import UspsDataset
from simple_tf.fashion_net.fashion_cign_lite import FashionCignLite
from simple_tf.fashion_net.fashion_cign_lite_early_exit import FashionCignLiteEarlyExit
from simple_tf.fashion_net.fashion_cign_moe_logits import FashionCignMoeLogits
from simple_tf.fashion_net.fashion_cign_random_sample import FashionCignRandomSample
from simple_tf.fashion_net.fashion_net_baseline import FashionNetBaseline
from simple_tf.fashion_net.fashion_net_single_late_exit import FashionNetSingleLateExit
from simple_tf.global_params import GlobalConstants
from auxillary.constants import DatasetTypes
from simple_tf.lenet.lenet_baseline import LeNetBaseline
from simple_tf.lenet.lenet_cign import LenetCign
from simple_tf.lenet.lenet_cign_early_exit import LenetCignEarlyExit
from simple_tf.lenet.lenet_random_sample import LenetCignRandomSample

use_moe = False
use_sampling = False
use_random_sampling = False
use_baseline = True
use_early_exit = False
use_late_exit = False


def get_network(dataset, network_name):
    if not (use_baseline or use_early_exit or use_late_exit or use_random_sampling):
        network = LenetCign(dataset=dataset, degree_list=GlobalConstants.TREE_DEGREE_LIST,
                            network_name="LeNetMNIST_CIGN")
    elif use_early_exit:
        network = LenetCignEarlyExit(dataset=dataset, degree_list=GlobalConstants.TREE_DEGREE_LIST,
                                     network_name="LeNetMNIST_CIGN_EarlyExit")
    elif use_random_sampling:
        network = LenetCignRandomSample(dataset=dataset, degree_list=GlobalConstants.TREE_DEGREE_LIST,
                                        network_name="LeNetMNIST_CIGN_RandomSample")
    else:
        network = LeNetBaseline(dataset=dataset, network_name="LeNetBaseline")
    return network


def usps_cign_training():
    network_name = "USPS_Net"
    dataset = UspsDataset(validation_sample_count=0)
    dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
    # Comparison with the NNRF algorithm
    NNRFComputationStatisticsCalculator.calculate(feature_size=dataset.trainingSamples.shape[-1],
                                                  class_count=dataset.get_label_count())
    classification_wd = [i * 0.00005 for i in range(0, 21)]
    decision_wd = [0.0]
    info_gain_balance_coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[classification_wd,
                                                                          decision_wd,
                                                                          info_gain_balance_coeffs])
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
        # network = get_network(dataset=dataset, network_name=network_name)
        # network.set_training_parameters()
        # network.build_network()
        # init = tf.global_variables_initializer()
        # print("********************NEW RUN:{0}********************".format(run_id))
        # network.set_hyperparameters(weight_decay_coefficient=tpl[0],
        #                             decision_weight_decay_coefficient=tpl[1],
        #                             info_gain_balance_coefficient=tpl[2],
        #                             classification_keep_probability=1.0,
        #                             decision_keep_probability=1.0,
        #                             early_exit_weight=1.0,
        #                             late_exit_weight=1.0)
        # experiment_id = DbLogger.get_run_id()
        # explanation = network.get_explanation_string()
        # series_id = 0
        # # series_id = int(run_id / GlobalConstants.EXPERIMENT_MULTIPLICATION_FACTOR)
        # explanation += "\n Series:{0}".format(series_id)
        # DbLogger.write_into_table(rows=[(experiment_id, explanation)], table=DbLogger.runMetaData, col_count=2)
        # sess.run(init)
        # network.train(sess=sess, dataset=dataset, run_id=experiment_id)
        # # try:
        # #     network.train(sess=sess, dataset=dataset, run_id=experiment_id)
        # # except Exception as e:
        # #     DbLogger.write_into_table(rows=[(experiment_id, -1, "Error", str(e))], table=DbLogger.runKvStore,
        # #                               col_count=4)
        # tf.reset_default_graph()
        # run_id += 1
