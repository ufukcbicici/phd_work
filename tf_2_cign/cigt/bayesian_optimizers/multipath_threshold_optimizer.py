import numpy as np
from sklearn.model_selection import train_test_split

from tf_2_cign.cigt.bayesian_optimizers.bayesian_optimizer import BayesianOptimizer

# Hyper-parameters
from tf_2_cign.utilities.utilities import Utilities


class MultipathThresholdOptimizer(BayesianOptimizer):
    def __init__(self, xi, init_points, n_iter,
                 model_id, val_ratio):
        super().__init__(xi, init_points, n_iter)
        self.model = None
        self.modelId = model_id
        self.valRatio = val_ratio
        self.routingProbabilities, self.routingEntropies, self.logits, self.groundTruths = self.get_model_outputs()
        probabilities_arr = list(self.routingProbabilities.values())[0]
        self.maxEntropies = []
        self.optimization_bounds_continuous = {}
        self.routingBlocksCount = 0
        for idx, arr in enumerate(probabilities_arr):
            self.routingBlocksCount += 1
            num_of_routes = arr.shape[1]
            self.maxEntropies.append(-np.log(1.0 / num_of_routes))
            self.optimization_bounds_continuous["entropy_block_{0}".format(idx)] = (0.0, self.maxEntropies[idx])
        self.totalSampleCount, self.valIndices, self.testIndices = self.prepare_val_test_sets()
        self.listOfEntropiesPerLevel = self.prepare_entropies_per_level_and_decision()
        self.routingCorrectnessDict = {}


    # Calculate entropies per level and per decision. The list by itself represents the levels.
    # Each element of the list is a numpy array, whose second and larger dimensions represent the
    # decisions taken in previous levels.
    def prepare_entropies_per_level_and_decision(self):
        decision_arrays = [[0, 1] for _ in range(self.routingBlocksCount)]
        decision_combinations = Utilities.get_cartesian_product(list_of_lists=decision_arrays)
        list_of_entropies_per_level = []

        for block_id in range(self.routingBlocksCount):
            num_of_decision_dimensions = 2 ** block_id
            entropy_array = np.zeros(shape=(self.totalSampleCount, num_of_decision_dimensions))
            list_of_entropies_per_level.append(entropy_array)

            all_previous_combinations = Utilities.get_cartesian_product(
                [[0, 1] for _ in range(block_id)])
            for previous_combination in all_previous_combinations:
                valid_combinations = []
                for combination in decision_combinations:
                    if combination[0:block_id] == previous_combination:
                        valid_combinations.append(combination)
                valid_probability_arrays = []
                valid_entropy_arrays = []
                for valid_combination in valid_combinations:
                    probability_array = self.routingProbabilities[valid_combination][block_id]
                    entropy_array = self.routingEntropies[valid_combination][block_id]
                    valid_probability_arrays.append(probability_array)
                    valid_entropy_arrays.append(entropy_array)

                # Assert that the result of same action sequences are always equal.
                for i_ in range(len(valid_probability_arrays) - 1):
                    assert np.allclose(valid_probability_arrays[i_], valid_probability_arrays[i_ + 1])
                for i_ in range(len(valid_entropy_arrays) - 1):
                    assert np.allclose(valid_entropy_arrays[i_], valid_entropy_arrays[i_ + 1])

                valid_entropies_matrix = np.stack(valid_entropy_arrays, axis=1)
                valid_entropy_arr = np.mean(valid_entropies_matrix, axis=1)
                if len(all_previous_combinations) == 1:
                    assert all_previous_combinations[0] == ()
                    list_of_entropies_per_level[block_id][:, 0] = valid_entropy_arr
                else:
                    combination_coord = int("".join(str(ele) for ele in previous_combination), 2)
                    list_of_entropies_per_level[block_id][:, combination_coord] = valid_entropy_arr
        return list_of_entropies_per_level

    def get_correctness_and_mac_dicts(self):
        correctness_dict = {}
        mac_dict = {}
        decision_arrays = [[0, 1] for _ in range(self.routingBlocksCount)]
        decision_combinations = Utilities.get_cartesian_product(list_of_lists=decision_arrays)
        for decision_combination in decision_combinations:
            correctness_dict[decision_combination] = []
            # Get the mac cost for this routing combination.
            combination_mac_cost = self.model.cigtNodes[0].macCost
            for idx, decision in enumerate(decision_combination):
                level_mac_cost = self.model.cigtNodes[idx].macCost
                if decision == 0:
                    combination_mac_cost += level_mac_cost
                else:
                    combination_mac_cost += self.model.pathCounts[idx + 1] * level_mac_cost
            mac_dict[decision_combination] = combination_mac_cost

            # Get correctness vectors.
            for idx in range(self.totalSampleCount):
                correct_label = self.groundTruths[decision_combination][idx]
                logits = self.logits[decision_combination][idx]
                estimated_label = np.argmax(logits)
                correctness_dict[decision_combination].append(int(correct_label == estimated_label))
        return correctness_dict

    def prepare_val_test_sets(self):
        total_sample_count = set()
        for ll in self.routingProbabilities.values():
            for arr in ll:
                total_sample_count.add(arr.shape[0])
        for ll in self.routingEntropies.values():
            for arr in ll:
                total_sample_count.add(arr.shape[0])
        for arr in self.logits.values():
            total_sample_count.add(arr.shape[0])
        for arr in self.groundTruths.values():
            total_sample_count.add(arr.shape[0])
        assert len(total_sample_count) == 1
        total_sample_count = list(total_sample_count)[0]
        val_sample_count = int(total_sample_count * self.valRatio)
        indices = np.arange(total_sample_count)
        val_indices, test_indices = train_test_split(indices, train_size=val_sample_count)
        return total_sample_count, val_indices, test_indices

    def get_model_outputs(self):
        return {}, {}, {}, {}

    def get_metrics(self, indices, thresholds):
        selections_arr = np.zeros(shape=(len(indices), self.routingBlocksCount), dtype=np.int32)
        selections_arr[:] = -10000
        for level in range(self.routingBlocksCount):
            threshold = thresholds[level]
            if level == 0:
                curr_entropies = self.listOfEntropiesPerLevel[level][indices][:, 0]
            else:
                selection_coords = np.apply_along_axis(func1d=lambda r: int("".join(str(ele) for ele in r), 2),
                                                       axis=1, arr=selections_arr[:, 0:level])
                curr_entropies = self.listOfEntropiesPerLevel[level][indices][np.arange(len(indices)), selection_coords]
            this_level_selections = np.array(curr_entropies >= threshold, dtype=np.int32)
            selections_arr[:, level] = this_level_selections



    def cost_function(self, **kwargs):
        thresholds = []
        for level in range(self.routingBlocksCount):
            thresholds.append(kwargs["entropy_block_{0}".format(level)])

        self.get_metrics(indices=self.valIndices, thresholds=thresholds)
        self.get_metrics(indices=self.testIndices, thresholds=thresholds)


        # X = kwargs["classification_dropout_probability"]
        # Y = self.information_gain_balance_coefficient # kwargs["information_gain_balance_coefficient"]
        # Z = self.decision_loss_coefficient # kwargs["decision_loss_coefficient"]
        # W = 0.01

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
