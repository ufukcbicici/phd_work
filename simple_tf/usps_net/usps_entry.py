import os

import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization

from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from data_handling.usps_dataset import UspsDataset
from simple_tf.global_params import GlobalConstants
from simple_tf.usps_net.usps_baseline import UspsBaseline

use_moe = False
use_sampling = False
use_random_sampling = False
use_baseline = True
use_early_exit = False
use_late_exit = False


def get_network(dataset, network_name):
    if not (use_baseline or use_early_exit or use_late_exit or use_random_sampling):
        raise NotImplementedError()
    elif use_early_exit:
        raise NotImplementedError()
    elif use_random_sampling:
        raise NotImplementedError()
    else:
        network = UspsBaseline(dataset=dataset, network_name=network_name)
    return network


def train_func(**kwargs):
    network_name = "USPS_CIGN_Baseline"
    dataset = UspsDataset(validation_sample_count=0)
    # Arriving from the Bayesian Optimization Step
    classification_wd = kwargs["classification_wd"]
    GlobalConstants.INITIAL_LR = kwargs["initial_lr"]
    # classification_wd = [i * 0.00005 for i in range(0, 21)]
    # decision_wd = [0.0]
    # info_gain_balance_coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    # cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[classification_wd,
    #                                                                       decision_wd,
    #                                                                       info_gain_balance_coeffs])

    dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
    # Session initialization
    if GlobalConstants.USE_CPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()
    network = get_network(dataset=dataset, network_name=network_name)
    network.set_training_parameters()
    network.build_network()
    init = tf.global_variables_initializer()
    print("********************NEW RUN:{0}********************")
    network.set_hyperparameters(weight_decay_coefficient=classification_wd,
                                decision_weight_decay_coefficient=0.0,
                                info_gain_balance_coefficient=1.0,
                                classification_keep_probability=1.0,
                                decision_keep_probability=1.0,
                                early_exit_weight=1.0,
                                late_exit_weight=1.0)
    experiment_id = DbLogger.get_run_id()
    explanation = network.get_explanation_string()
    series_id = 0
    # series_id = int(run_id / GlobalConstants.EXPERIMENT_MULTIPLICATION_FACTOR)
    explanation += "\n Series:{0}".format(series_id)
    DbLogger.write_into_table(rows=[(experiment_id, explanation)], table=DbLogger.runMetaData, col_count=2)
    sess.run(init)
    train_accuracies, validation_accuracies = network.train(sess=sess, dataset=dataset, run_id=experiment_id)
    mean_val_accuracy = np.mean(np.array(validation_accuracies[-10:]))
    tf.reset_default_graph()
    return mean_val_accuracy


def usps_cign_training():
    pbounds = {"classification_wd": (0.0, 0.001), "initial_lr": (0.0001, 0.1)}
    train_func(classification_wd=0.0004, initial_lr=0.001)

    # optimizer = BayesianOptimization(
    #     f=train_func,
    #     pbounds=pbounds,
    # )
    # optimizer.maximize(
    #     init_points=25,
    #     n_iter=50,
    #     acq="ei",
    #     xi=0.0
    # )
    print("X")
