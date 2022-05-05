import tensorflow as tf
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
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

optimization_bounds_continuous = {
    "classification_dropout_probability": (0.0, 0.5),
    "information_gain_balance_coefficient": (1.0, 10.0),
    "decision_loss_coefficient": (0.01, 1.0),
    "lr_initial_rate": (0.0, 0.05)
}

optimization_bounds_discrete = {
    "classification_dropout_probability": (0.0, 1000.0),
    "information_gain_balance_coefficient": (0.0, 1000.0),
    "decision_loss_coefficient": (0.0, 1000.0)
}

discrete_values = {
    "classification_dropout_probability": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    "information_gain_balance_coefficient": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
                                             8.5, 9.0, 9.5, 10.0],
    "decision_loss_coefficient": [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                  0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
}


def cigt_test_function(classification_dropout_probability,
                       information_gain_balance_coefficient,
                       decision_loss_coefficient,
                       lr_initial_rate):
    X = classification_dropout_probability
    Y = information_gain_balance_coefficient
    Z = decision_loss_coefficient
    W = lr_initial_rate

    print("classification_dropout_probability={0}".format(classification_dropout_probability))
    print("information_gain_balance_coefficient={0}".format(information_gain_balance_coefficient))
    print("decision_loss_coefficient={0}".format(decision_loss_coefficient))
    print("lr_initial_rate={0}".format(lr_initial_rate))
    #
    # dX = X - 0.25
    # dY = Y - 5.5
    # dZ = Z - (0.01 + (1.0 - 0.01) / 2.0)
    #
    # score = -(dX * dX + dY * dY * dZ * dZ)

    fashion_mnist = FashionMnist(batch_size=FashionNetConstants.batch_size, validation_size=0)
    softmax_decay_controller = StepWiseDecayAlgorithm(decay_name="Stepwise",
                                                      initial_value=FashionNetConstants.softmax_decay_initial,
                                                      decay_coefficient=FashionNetConstants.softmax_decay_coefficient,
                                                      decay_period=FashionNetConstants.softmax_decay_period,
                                                      decay_min_limit=FashionNetConstants.softmax_decay_min_limit)
    learning_rate_calculator = DiscreteParameter(name="lr_calculator",
                                                 value=W,
                                                 schedule=[(15000 + 12000, (1.0 / 2.0) * W),
                                                           (30000 + 12000, (1.0 / 4.0) * W),
                                                           (40000 + 12000, (1.0 / 40.0) * W)])
    print(learning_rate_calculator)

    with tf.device("GPU"):
        run_id = DbLogger.get_run_id()
        fashion_cigt = LenetCigt(batch_size=125,
                                 input_dims=(28, 28, 1),
                                 filter_counts=[32, 64, 128],
                                 kernel_sizes=[5, 5, 1],
                                 hidden_layers=[512, 256],
                                 decision_drop_probability=0.0,
                                 classification_drop_probability=X,
                                 decision_wd=0.0,
                                 classification_wd=0.0,
                                 decision_dimensions=[128, 128],
                                 class_count=10,
                                 information_gain_balance_coeff=Y,
                                 softmax_decay_controller=softmax_decay_controller,
                                 learning_rate_schedule=learning_rate_calculator,
                                 decision_loss_coeff=Z,
                                 path_counts=[2, 4],
                                 bn_momentum=0.9,
                                 warm_up_period=25,
                                 routing_strategy_name="Approximate_Training",
                                 run_id=run_id,
                                 evaluation_period=10,
                                 measurement_start=25)

        explanation = fashion_cigt.get_explanation_string()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
        score = fashion_cigt.fit(x=fashion_mnist.trainDataTf, validation_data=fashion_mnist.testDataTf,
                                 epochs=FashionNetConstants.epoch_count)

    return score


def cigt_test_function_discretized(classification_dropout_probability,
                                   information_gain_balance_coefficient,
                                   decision_loss_coefficient):
    X = Utilities.discretize_value(sampled_value=classification_dropout_probability,
                                   interval_start=optimization_bounds_discrete["classification_dropout_probability"][0],
                                   interval_end=optimization_bounds_discrete["classification_dropout_probability"][1],
                                   discrete_values=discrete_values["classification_dropout_probability"])
    Y = Utilities.discretize_value(sampled_value=information_gain_balance_coefficient,
                                   interval_start=optimization_bounds_discrete[
                                       "information_gain_balance_coefficient"][0],
                                   interval_end=optimization_bounds_discrete["information_gain_balance_coefficient"][1],
                                   discrete_values=discrete_values["information_gain_balance_coefficient"])
    Z = Utilities.discretize_value(sampled_value=decision_loss_coefficient,
                                   interval_start=optimization_bounds_discrete[
                                       "decision_loss_coefficient"][0],
                                   interval_end=optimization_bounds_discrete["decision_loss_coefficient"][1],
                                   discrete_values=discrete_values["decision_loss_coefficient"])

    dX = -(X - 0.35) ** 2 + np.random.normal(loc=0.0, scale=0.1)
    dY = -(Y - 0.2) ** 2 + np.random.normal(loc=0.0, scale=0.1)
    dZ = -(Z - 0.5) ** 2 + np.random.normal(loc=0.0, scale=0.1)

    score = dX + dY + dZ
    return score


def optimize_with_bayesian_optimization():
    optimizer = BayesianOptimization(
        f=cigt_test_function,
        pbounds=optimization_bounds_continuous,
        verbose=10
    )

    logger = JSONLogger(path="bo_logs_with_lr_optimization.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        n_iter=300,
        init_points=100,
        acq="ei",
        xi=0.01)


def optimize_with_discretized_bayesian_optimization():
    optimizer = BayesianOptimization(
        f=cigt_test_function_discretized,
        pbounds=optimization_bounds_discrete,
        verbose=10
    )

    logger = JSONLogger(path="bo_logs_discrete.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        n_iter=300,
        init_points=100,
        acq="ei",
        xi=0.01)

    print("X")
