import tensorflow as tf
import numpy as np
from bayes_opt import BayesianOptimization

from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cign import Cign
from tf_2_cign.cigt.cigt import Cigt
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.fashion_net.fashion_cign import FashionCign
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants
from tf_2_cign.utilities.utilities import Utilities

# Hyper-parameters
from tf_2_cign.fashion_net.fashion_cign_rl import FashionCignRl
from tf_2_cign.fashion_net.fashion_cign_binary_rl import FashionRlBinaryRouting
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm


def cigt_test_function(classification_dropout_probability,
                       information_gain_balance_coefficient,
                       decision_loss_coefficient):
    X = classification_dropout_probability
    Y = information_gain_balance_coefficient
    Z = decision_loss_coefficient
    fashion_mnist = FashionMnist(batch_size=FashionNetConstants.batch_size, validation_size=0)
    softmax_decay_controller = StepWiseDecayAlgorithm(decay_name="Stepwise",
                                                      initial_value=FashionNetConstants.softmax_decay_initial,
                                                      decay_coefficient=FashionNetConstants.softmax_decay_coefficient,
                                                      decay_period=FashionNetConstants.softmax_decay_period,
                                                      decay_min_limit=FashionNetConstants.softmax_decay_min_limit)
    with tf.device("GPU"):
        run_id = DbLogger.get_run_id()
        fashion_cigt = LenetCigt(batch_size=FashionNetConstants.batch_size,
                                 input_dims=FashionNetConstants.input_dims,
                                 filter_counts=FashionNetConstants.filter_counts,
                                 kernel_sizes=FashionNetConstants.kernel_sizes,
                                 hidden_layers=FashionNetConstants.hidden_layers,
                                 decision_drop_probability=FashionNetConstants.decision_drop_probability,
                                 classification_drop_probability=X,
                                 decision_wd=FashionNetConstants.decision_wd,
                                 classification_wd=FashionNetConstants.classification_wd,
                                 decision_dimensions=FashionNetConstants.decision_dimensions,
                                 class_count=10,
                                 information_gain_balance_coeff=Y,
                                 softmax_decay_controller=softmax_decay_controller,
                                 learning_rate_schedule=FashionNetConstants.learning_rate_calculator,
                                 decision_loss_coeff=Z,
                                 path_counts=FashionNetConstants.path_counts,
                                 bn_momentum=FashionNetConstants.bn_momentum,
                                 warm_up_period=FashionNetConstants.warm_up_period,
                                 routing_strategy_name="Approximate_Training",
                                 run_id=run_id,
                                 evaluation_period=10,
                                 measurement_start=11)
        explanation = fashion_cigt.get_explanation_string()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
        score = fashion_cigt.fit(x=fashion_mnist.trainDataTf, validation_data=fashion_mnist.testDataTf,
                                 epochs=FashionNetConstants.epoch_count)
        return score


def optimize_with_bayesian_optimization():
    pbounds = {
        "classification_dropout_probability": (0.0, 0.5),
        "information_gain_balance_coefficient": (1.0, 10.0),
        "decision_loss_coefficient": (0.01, 1.0)
    }

    optimizer = BayesianOptimization(
        f=cigt_test_function,
        pbounds=pbounds
    )

    optimizer.maximize(
        n_iter=1000,
        init_points=300,
        acq="ei",
        xi=0.01)
