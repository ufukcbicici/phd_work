import os

import tensorflow as tf

from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from data_handling.cifar_dataset import CifarDataSet
from simple_tf.cifar_nets.cifar100_cign import Cifar100_Cign
from simple_tf.cifar_nets.cifar100_cign_random_sampling import Cifar100_CignRandomSampling
from simple_tf.cifar_nets.cifar100_cign_sampling import Cifar100_CignSampling
from simple_tf.cifar_nets.cifar100_multi_gpu_cign import Cifar100_MultiGpuCign
from simple_tf.cifar_nets.cifar100_resnet_baseline import Cifar100_Baseline
from simple_tf.global_params import GlobalConstants

use_multi_gpu = False
use_sampling = False
use_random_sampling = False
use_baseline = False


def get_network(dataset):
    if not use_multi_gpu and not use_sampling and not use_random_sampling and not use_baseline:
        network = Cifar100_Cign(degree_list=GlobalConstants.TREE_DEGREE_LIST, dataset=dataset)
    elif use_multi_gpu:
        network = Cifar100_MultiGpuCign(degree_list=GlobalConstants.TREE_DEGREE_LIST, dataset=dataset)
    elif use_sampling:
        network = Cifar100_CignSampling(degree_list=GlobalConstants.TREE_DEGREE_LIST, dataset=dataset)
    elif use_random_sampling:
        network = Cifar100_CignRandomSampling(degree_list=GlobalConstants.TREE_DEGREE_LIST, dataset=dataset)
    elif use_baseline:
        network = Cifar100_Baseline(dataset=dataset)
    else:
        raise NotImplementedError()
    return network


def cifar_100_training():
    import sys
    print(sys.version)
    classification_wd = [0.0002]
    # classification_wd.extend([0.0004, 0.00045, 0.0005, 0.00055, 0.0006] * GlobalConstants.EXPERIMENT_MULTIPLICATION_FACTOR)
    classification_wd = sorted(classification_wd)
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
        dataset = CifarDataSet(session=sess, validation_sample_count=0, load_validation_from=None)
        dataset.set_curr_session(sess=sess)
        network = get_network(dataset=dataset)
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
        dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
        network.train(sess=sess, dataset=dataset, run_id=experiment_id)
        tf.reset_default_graph()
        run_id += 1
