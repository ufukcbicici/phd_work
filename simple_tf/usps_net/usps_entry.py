import os

import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization

from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from data_handling.usps_dataset import UspsDataset
from simple_tf.global_params import GlobalConstants
from simple_tf.usps_net.usps_baseline import UspsBaseline
from simple_tf.usps_net.usps_cign import UspsCIGN
from simple_tf.usps_net.usps_random_sample import UspsCIGNRandomSample

use_moe = False
use_sampling = False
use_random_sampling = False
use_baseline = False
use_early_exit = False
use_late_exit = False


def get_network(dataset, network_name):
    if not (use_baseline or use_early_exit or use_late_exit or use_random_sampling):
        network = UspsCIGN(dataset=dataset, network_name=network_name, degree_list=GlobalConstants.TREE_DEGREE_LIST)
    elif use_early_exit:
        raise NotImplementedError()
    elif use_random_sampling:
        network = UspsCIGNRandomSample(dataset=dataset, network_name=network_name,
                                       degree_list=GlobalConstants.TREE_DEGREE_LIST)
    else:
        network = UspsBaseline(dataset=dataset, network_name=network_name)
    return network


def train_func(**kwargs):
    network_name = "USPS_CIGN"
    dataset = UspsDataset(validation_sample_count=0)
    # Arriving from the Bayesian Optimization Step
    classification_wd = kwargs["classification_wd"]
    decision_wd = 0.0
    info_gain_balance_coefficient = 1.0  # kwargs["info_gain_balance_coefficient"]
    GlobalConstants.INITIAL_LR = kwargs["initial_lr"]
    UspsCIGN.SOFTMAX_DECAY_INITIAL = kwargs["softmax_decay_initial"]
    UspsCIGN.SOFTMAX_DECAY_PERIOD = int(kwargs["softmax_decay_period"])
    UspsCIGN.THRESHOLD_LOWER_LIMIT = kwargs["threshold_lower_limit"]
    UspsCIGN.THRESHOLD_PERIOD = int(kwargs["threshold_period"])

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
                                decision_weight_decay_coefficient=decision_wd,
                                info_gain_balance_coefficient=info_gain_balance_coefficient,
                                # THESE REST ARE IRRELEVANT
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
    pbounds = {"classification_wd": (0.0, 0.001),
               "initial_lr": (0.0001, 0.1),
               "softmax_decay_initial": (1.0, 50.0),
               "softmax_decay_period": (100.0, 5000.0),
               "threshold_lower_limit": (0.0, 0.5),
               "threshold_period": (100.0, 5000.0)}
    # "info_gain_balance_coefficient": (1.0, 5.0)}

    # Best Pairs
    # best_hyperparameter_pairs = [(0.06, 0.00146918),
    #                              (0.06, 0.00584801),
    #                              (0.06, 0.00163591)]
    # experiment_count_per_params = 25
    # best_hyperparameter_pairs = experiment_count_per_params * best_hyperparameter_pairs
    # for param_tpl in best_hyperparameter_pairs:
    #     initial_lr = param_tpl[0]
    #     classification_wd = param_tpl[1]
    #     train_func(classification_wd=classification_wd, initial_lr=initial_lr)

    optimizer = BayesianOptimization(
        f=train_func,
        pbounds=pbounds,
    )
    optimizer.maximize(
        init_points=100,
        n_iter=500,
        acq="ei",
        xi=0.0
    )
    print("X")
