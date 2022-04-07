import tensorflow as tf

from auxillary.db_logger import DbLogger
from tf_2_cign.utilities.utilities import Utilities
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.data.fashion_mnist import FashionMnist
# Hyper-parameters
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # classification_wd = [0.0]
    # decision_wd = [0.0]
    # info_gain_balance_coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    # # # classification_dropout_probs = [0.15]
    # classification_dropout_probs = sorted([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] *
    #                                       GlobalConstants.EXPERIMENT_MULTIPLICATION_FACTOR)
    # # info_gain_balance_coeffs = [1.0]
    # # classification_dropout_probs = [0.15]
    # # classification_dropout_probs = sorted([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    # #                                       * GlobalConstants.EXPERIMENT_MULTIPLICATION_FACTOR)
    # # classification_dropout_probs = [0.0]
    # # decision_dropout_probs = [0.0]
    # cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=[classification_wd,
    #                                                                       decision_wd,
    #                                                                       info_gain_balance_coeffs,
    #                                                                       classification_dropout_probs,
    #                                                                       decision_dropout_probs])

    info_gain_balance_coeffs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    classification_dropout_probs = sorted([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] *
                                          FashionNetConstants.experiment_factor)

    cartesian_product = Utilities.get_cartesian_product(list_of_lists=[info_gain_balance_coeffs,
                                                                       classification_dropout_probs])
    run_id = 0
    for tpl in cartesian_product:
        fashion_mnist = FashionMnist(batch_size=FashionNetConstants.batch_size,
                                     validation_size=0)
        softmax_decay_controller = StepWiseDecayAlgorithm(
            decay_name="Stepwise",
            initial_value=FashionNetConstants.softmax_decay_initial,
            decay_coefficient=FashionNetConstants.softmax_decay_coefficient,
            decay_period=FashionNetConstants.softmax_decay_period,
            decay_min_limit=FashionNetConstants.softmax_decay_min_limit)
        learning_rate_calculator = DiscreteParameter(name="lr_calculator",
                                                     value=0.01,
                                                     schedule=[(15000 + 12000, 0.005),
                                                               (30000 + 12000, 0.0025),
                                                               (40000 + 12000, 0.00025)])
        info_gain_balance_coeff = tpl[0]
        classification_dropout_prob = tpl[1]
        warm_up_period = 25

        with tf.device("GPU"):
            run_id = DbLogger.get_run_id()
            fashion_cigt = LenetCigt(batch_size=125,
                                     input_dims=(28, 28, 1),
                                     filter_counts=[32, 64, 128],
                                     kernel_sizes=[5, 5, 1],
                                     hidden_layers=[256, 128],
                                     decision_drop_probability=0.0,
                                     classification_drop_probability=classification_dropout_prob,
                                     decision_wd=0.0,
                                     classification_wd=0.0,
                                     decision_dimensions=[128, 128],
                                     class_count=10,
                                     information_gain_balance_coeff=info_gain_balance_coeff,
                                     softmax_decay_controller=softmax_decay_controller,
                                     learning_rate_schedule=learning_rate_calculator,
                                     decision_loss_coeff=1.0,
                                     path_counts=[2, 2],
                                     bn_momentum=0.9,
                                     warm_up_period=warm_up_period,
                                     routing_strategy_name="Approximate_Training",
                                     run_id=run_id,
                                     evaluation_period=10)
        explanation = fashion_cigt.get_explanation_string()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
        fashion_cigt.fit(x=fashion_mnist.trainDataTf, validation_data=fashion_mnist.testDataTf, epochs=125)
