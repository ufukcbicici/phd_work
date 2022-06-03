import tensorflow as tf

from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.bayesian_optimizers.bayesian_optimizer import BayesianOptimizer

# Hyper-parameters
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants


class FmnistGumbelSoftmaxOnlyDropoutOptimizer(BayesianOptimizer):
    def __init__(self, xi, init_points, n_iter, ig_balance_coeff, d_loss_coeff):
        super().__init__(xi, init_points, n_iter)
        self.information_gain_balance_coefficient = ig_balance_coeff
        self.decision_loss_coefficient = d_loss_coeff
        self.optimization_bounds_continuous = {
            "classification_dropout_probability": (0.0, 0.5)
        }

    def cost_function(self, **kwargs):
        # lr_initial_rate,
        # hyperbolic_exponent):
        X = kwargs["classification_dropout_probability"]
        Y = self.information_gain_balance_coefficient # kwargs["information_gain_balance_coefficient"]
        Z = self.decision_loss_coefficient # kwargs["decision_loss_coefficient"]
        W = 0.01

        print("classification_dropout_probability={0}".format(kwargs["classification_dropout_probability"]))
        # print("information_gain_balance_coefficient={0}".format(kwargs["information_gain_balance_coefficient"]))
        # print("decision_loss_coefficient={0}".format(kwargs["decision_loss_coefficient"]))
        # print("lr_initial_rate={0}".format(kwargs["lr_initial_rate"]))

        FashionNetConstants.softmax_decay_initial = 25.0
        FashionNetConstants.softmax_decay_coefficient = 0.9999
        FashionNetConstants.softmax_decay_period = 1
        FashionNetConstants.softmax_decay_min_limit = 0.1

        fashion_mnist = FashionMnist(batch_size=FashionNetConstants.batch_size, validation_size=0)
        softmax_decay_controller = StepWiseDecayAlgorithm(
            decay_name="Stepwise",
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
                                     routing_strategy_name="Full_Training",
                                     run_id=run_id,
                                     evaluation_period=10,
                                     measurement_start=25,
                                     use_straight_through=True,
                                     optimizer_type="SGD",
                                     decision_non_linearity="Softmax",
                                     save_model=True,
                                     model_definition="Lenet CIGT - Gumbel Softmax with E[Z] based routing - Softmax and Straight Through Bayesian Optimization - Only Dropout Optimization")

            explanation = fashion_cigt.get_explanation_string()
            DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
            score = fashion_cigt.fit(x=fashion_mnist.trainDataTf, validation_data=fashion_mnist.testDataTf,
                                     epochs=FashionNetConstants.epoch_count)

        return score
