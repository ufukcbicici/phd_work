import numpy as np
from tf_2_cign.cigt.bayesian_optimizers.bayesian_optimizer import BayesianOptimizer

# Hyper-parameters


class MultipathThresholdOptimizer(BayesianOptimizer):
    def __init__(self, xi, init_points, n_iter,
                 model_id, val_ratio):
        super().__init__(xi, init_points, n_iter)
        self.modelId = model_id
        self.valRatio = val_ratio
        self.routingProbabilities, self.routingEntropies, self.logits, self.groundTruths = self.get_model_outputs()
        probabilities_arr = list(self.routingProbabilities.values())[0]
        self.maxEntropies = []
        self.optimization_bounds_continuous = {}
        for idx, arr in enumerate(probabilities_arr):
            self.maxEntropies.append(-np.log(1.0 / arr.shape[0]))
            self.optimization_bounds_continuous["entropy_block_{0}".format(idx)] = (0.0, self.maxEntropies[idx])
        print("X")

    def get_model_outputs(self):
        return {}, {}, {}, {}

    def cost_function(self, **kwargs):
        return 0
        # # lr_initial_rate,
        # # hyperbolic_exponent):
        # X = kwargs["classification_dropout_probability"]
        # Y = kwargs["information_gain_balance_coefficient"]
        # Z = kwargs["decision_loss_coefficient"]
        # W = 0.01
        #
        # print("classification_dropout_probability={0}".format(kwargs["classification_dropout_probability"]))
        # print("information_gain_balance_coefficient={0}".format(kwargs["information_gain_balance_coefficient"]))
        # print("decision_loss_coefficient={0}".format(kwargs["decision_loss_coefficient"]))
        # # print("lr_initial_rate={0}".format(kwargs["lr_initial_rate"]))
        #
        # FashionNetConstants.softmax_decay_initial = 25.0
        # FashionNetConstants.softmax_decay_coefficient = 0.9999
        # FashionNetConstants.softmax_decay_period = 1
        # FashionNetConstants.softmax_decay_min_limit = 0.1
        #
        # fashion_mnist = FashionMnist(batch_size=FashionNetConstants.batch_size, validation_size=0)
        # softmax_decay_controller = StepWiseDecayAlgorithm(
        #     decay_name="Stepwise",
        #     initial_value=FashionNetConstants.softmax_decay_initial,
        #     decay_coefficient=FashionNetConstants.softmax_decay_coefficient,
        #     decay_period=FashionNetConstants.softmax_decay_period,
        #     decay_min_limit=FashionNetConstants.softmax_decay_min_limit)
        #
        # learning_rate_calculator = DiscreteParameter(name="lr_calculator",
        #                                              value=W,
        #                                              schedule=[(15000 + 12000, (1.0 / 2.0) * W),
        #                                                        (30000 + 12000, (1.0 / 4.0) * W),
        #                                                        (40000 + 12000, (1.0 / 40.0) * W)])
        # print(learning_rate_calculator)
        #
        # with tf.device("GPU"):
        #     run_id = DbLogger.get_run_id()
        #     fashion_cigt = LenetCigt(batch_size=125,
        #                              input_dims=(28, 28, 1),
        #                              filter_counts=[32, 64, 128],
        #                              kernel_sizes=[5, 5, 1],
        #                              hidden_layers=[512, 256],
        #                              decision_drop_probability=0.0,
        #                              classification_drop_probability=X,
        #                              decision_wd=0.0,
        #                              classification_wd=0.0,
        #                              decision_dimensions=[128, 128],
        #                              class_count=10,
        #                              information_gain_balance_coeff=Y,
        #                              softmax_decay_controller=softmax_decay_controller,
        #                              learning_rate_schedule=learning_rate_calculator,
        #                              decision_loss_coeff=Z,
        #                              path_counts=[2, 4],
        #                              bn_momentum=0.9,
        #                              warm_up_period=25,
        #                              routing_strategy_name="Full_Training",
        #                              run_id=run_id,
        #                              evaluation_period=10,
        #                              measurement_start=25,
        #                              use_straight_through=True,
        #                              optimizer_type="SGD",
        #                              decision_non_linearity="Softmax",
        #                              save_model=True,
        #                              model_definition="Lenet CIGT - Gumbel Softmax with E[Z] based routing - Softmax and Straight Through Bayesian Optimization")
        #
        #     explanation = fashion_cigt.get_explanation_string()
        #     DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData)
        #     score = fashion_cigt.fit(x=fashion_mnist.trainDataTf, validation_data=fashion_mnist.testDataTf,
        #                              epochs=FashionNetConstants.epoch_count)
        #
        # return score
