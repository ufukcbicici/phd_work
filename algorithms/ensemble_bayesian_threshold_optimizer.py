import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization
from sklearn.metrics import f1_score
from algorithms.information_gain_routing_accuracy_calculator import InformationGainRoutingAccuracyCalculator
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer import \
    DirectThresholdOptimizer
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer_entropy import \
    DirectThresholdOptimizerEntropy
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs
from algorithms.bayesian_threshold_optimizer import BayesianThresholdOptimizer


class EnsembleBayesianThresholdOptimizer(BayesianThresholdOptimizer):
    def __init__(self,
                 list_of_networks, list_of_routing_data, list_of_temperature_dicts,
                 session, seed, threshold_kind, mixing_lambda, run_id):
        self.listOfNetworks = list_of_networks
        self.listOfRoutingData = list_of_routing_data
        assert len(list_of_networks) == len(list_of_routing_data)
        self.ensembleSize = len(self.listOfNetworks)
        self.session = session
        self.seed = seed
        self.runId = run_id
        self.listOfTemperatureDicts = list_of_temperature_dicts
        self.thresholdKind = threshold_kind
        self.mixingLambda = mixing_lambda
        self.thresholdOptimizers = []
        for network_id in range(self.ensembleSize):
            with tf.variable_scope("network_{0}".format(network_id)):
                if self.thresholdKind == "entropy":
                    threshold_optimizer = DirectThresholdOptimizerEntropy(
                        network=self.listOfNetworks[network_id],
                        routing_data=self.listOfRoutingData[network_id],
                        seed=self.seed,
                        use_parametric_weights=True)
                else:
                    threshold_optimizer = DirectThresholdOptimizer(
                        network=self.listOfNetworks[network_id],
                        routing_data=self.listOfRoutingData[network_id],
                        seed=self.seed,
                        use_parametric_weights=True)
                self.thresholdOptimizers.append(threshold_optimizer)
                threshold_optimizer.build_network()
        self.fixedThresholds = []
        self.fixedWeights = []
        self.listOfResults = []

    def get_random_thresholds(self):
        list_of_thresholds = []
        for network in self.listOfNetworks:
            thresholds_dict = {}
            for node in network.innerNodes:
                child_nodes = network.dagObject.children(node)
                child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
                child_count = len(child_nodes)
                if self.thresholdKind == "probabiliy":
                    max_bound = 1.0 / child_count
                    thresholds_dict[node.index] = np.random.uniform(low=0.0, high=max_bound, size=(1, child_count))
                elif self.thresholdKind == "entropy":
                    max_bound = -np.log(1.0 / child_count)
                    thresholds_dict[node.index] = np.random.uniform(low=0.0, high=max_bound)
                else:
                    raise NotImplementedError()
            list_of_thresholds.append(thresholds_dict)
        return list_of_thresholds

    def calculate_threshold_bounds(self):
        pbounds = {}
        for network_id, network in enumerate(self.listOfNetworks):
            for node in network.innerNodes:
                child_nodes = network.dagObject.children(node)
                child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
                if self.thresholdKind == "probability":
                    max_bound = 1.0 / len(child_nodes)
                    for c_nd in child_nodes:
                        pbounds["n_{0}_t_{1}_{2}".format(network_id, node.index, c_nd.index)] = (0.0, max_bound)
                elif self.thresholdKind == "entropy":
                    max_bound = -np.log(1.0 / len(child_nodes))
                    pbounds["n_{0}_t_{1}".format(network_id, node.index)] = (0.0, max_bound)
                    # if node.isRoot:
                    #     interval_size = max_bound / cluster_count
                    #     pbounds["c_{0}_t_{1}".format(cluster_id, node.index)] = (cluster_id*interval_size,
                    #                                                             (cluster_id+1)*interval_size)
                    # else:
                    #     pbounds["c_{0}_t_{1}".format(cluster_id, node.index)] = (0.0, max_bound)
                else:
                    raise NotImplementedError()
        return pbounds

    def calculate_weight_bounds(self, min_boundary, max_boundary):
        pbounds = {}
        for network_id, network in enumerate(self.listOfNetworks):
            for node in network.leafNodes:
                pbounds["n_{0}_leaf_{1}".format(network_id, node.index)] = (min_boundary, max_boundary)
        return pbounds

    def decode_threshold_parameters(self, args_dict):
        list_of_threshold_dicts = []
        for network_id in range(self.ensembleSize):
            thrs_dict = {}
            network = self.listOfNetworks[network_id]
            for node in network.innerNodes:
                child_nodes = network.dagObject.children(node)
                child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
                if self.thresholdKind == "probability":
                    thrs_arr = np.array([args_dict["n_{0}_t_{1}_{2}".format(network_id, node.index, c_nd.index)]
                                         for c_nd in child_nodes])
                    thrs_dict[node.index] = thrs_arr[np.newaxis, :]
                elif self.thresholdKind == "entropy":
                    thrs_dict[node.index] = args_dict["n_{0}_t_{1}".format(network_id, node.index)]
                else:
                    raise NotImplementedError()
            list_of_threshold_dicts.append(thrs_dict)
        return list_of_threshold_dicts

    def decode_weight_parameters(self, args_dict):
        list_of_weight_lists = []
        for network_id in range(self.ensembleSize):
            network = self.listOfNetworks[network_id]
            weight_list = []
            for node in network.leafNodes:
                weight = args_dict["n_{0}_leaf_{1}".format(network_id, node.index)]
                weight_list.append(weight)
            weight_list = np.array(weight_list)[np.newaxis, :]
            list_of_weight_lists.append(weight_list)
        return list_of_weight_lists

    def get_thresholding_results_for_args(self, kwargs):
        # Convert Bayesian Optimization space sample into usable thresholds
        list_of_threshold_dicts = self.decode_threshold_parameters(kwargs) \
            if self.fixedThresholds is None else self.fixedThresholds
        list_of_weight_lists = self.decode_weight_parameters(kwargs) if self.fixedWeights is None else self.fixedWeights
        list_of_results = []
        dict_of_posteriors = {"train": [], "test": []}
        dict_of_activation_costs = {"train": [], "test": []}
        dict_of_scores = {"train": [], "test": []}
        dict_of_labels = {"train": [], "test": []}
        for network_id in range(self.ensembleSize):
            routing_data = self.listOfRoutingData[network_id]
            train_indices = routing_data.trainingIndices
            test_indices = routing_data.testIndices
            sample_indices = {"train": train_indices, "test": test_indices}
            results_dict = {}
            for data_type in ["train", "test"]:
                results = self.get_thresholding_results(
                    indices=sample_indices[data_type],
                    thresholds_dict=list_of_threshold_dicts[network_id],
                    routing_weights_array=list_of_weight_lists[network_id],
                    temperatures_dict=self.listOfTemperatureDicts[network_id],
                    mixing_lambda=self.mixingLambda)
                results_dict[data_type] = results
                print("NetworkId:{0} {1} Accuracy: {2} {1} Computation Load:{3} {1} Score:{4}".format(
                    network_id,
                    data_type,
                    results_dict[data_type]["final_accuracy"],
                    results_dict[data_type]["final_activation_cost"],
                    results_dict[data_type]["final_score"]))
                dict_of_posteriors[data_type].append(results_dict[data_type]["final_posteriors"])
                dict_of_activation_costs[data_type].append(results_dict[data_type]["activation_costs_vector"])
                dict_of_scores[data_type].append(results_dict[data_type]["score_vector"])
                dict_of_labels[data_type].append(results_dict[data_type]["gt_labels"])
            list_of_results.append(results_dict)
        # Calculate ensemble results
        ensemble_results_dict = {}
        for data_type in ["train", "test"]:
            # Be sure that the labels are equal
            for network_id in range(self.ensembleSize - 1):
                assert np.array_equal(dict_of_labels[data_type][network_id], dict_of_labels[data_type][network_id + 1])
            true_labels = dict_of_labels[data_type][0]
            ensemble_posterior_tensor = np.stack(dict_of_posteriors[data_type], axis=-1)
            ensemble_posterior = np.mean(ensemble_posterior_tensor, axis=-1)
            ensemble_predicted_labels = np.argmax(ensemble_posterior, axis=-1)
            ensemble_correctness_vector = (ensemble_predicted_labels == true_labels).astype(np.float32)
            ensemble_activation_cost_vector = np.mean(np.stack(dict_of_activation_costs[data_type], axis=-1), axis=-1)
            ensemble_score_vector = self.mixingLambda * ensemble_correctness_vector - \
                                    (1.0 - self.mixingLambda) * ensemble_activation_cost_vector
            ensemble_final_accuracy = np.mean(ensemble_correctness_vector)
            ensemble_final_activation_cost = np.mean(ensemble_activation_cost_vector)
            ensemble_final_score = np.mean(ensemble_score_vector)
            ensemble_f1_macro = f1_score(y_true=true_labels, y_pred=ensemble_predicted_labels, average="macro")
            ensemble_f1_micro = f1_score(y_true=true_labels, y_pred=ensemble_predicted_labels, average="micro")
            ensemble_results_dict[data_type] = {"final_accuracy": ensemble_final_accuracy,
                                                "f1_macro": ensemble_f1_macro,
                                                "f1_micro": ensemble_f1_micro,
                                                "final_activation_cost": ensemble_final_activation_cost,
                                                "final_score": ensemble_final_score}
            print("{0}: Ensemble Accuracy:{1} Ensemble F1 Macro:{2} "
                  "Ensemble F1 Micro:{3} Computation Load:{4} Score:{5}".format(data_type,
                                                                                ensemble_final_accuracy,
                                                                                ensemble_f1_macro,
                                                                                ensemble_f1_micro,
                                                                                ensemble_final_activation_cost,
                                                                                ensemble_final_score))
        return list_of_results, ensemble_results_dict

    def loss_function(self, **kwargs):
        list_of_results, ensemble_results_dict = self.get_thresholding_results_for_args(kwargs)
        self.listOfResults.append(ensemble_results_dict)
        return ensemble_results_dict["train"]["final_score"]

    def calculate_ig_accuracies(self):
        train_indices = self.routingData.trainingIndices
        test_indices = self.routingData.testIndices
        self.fixedWeights = None
        self.fixedThresholds = None
        # Learn the standard information gain based accuracies
        train_ig_accuracy = InformationGainRoutingAccuracyCalculator.calculate(network=self.network,
                                                                               routing_data=self.routingData,
                                                                               indices=train_indices)
        test_ig_accuracy = InformationGainRoutingAccuracyCalculator.calculate(network=self.network,
                                                                              routing_data=self.routingData,
                                                                              indices=test_indices)
        print("train_ig_accuracy={0}".format(train_ig_accuracy))
        print("test_ig_accuracy={0}".format(test_ig_accuracy))

    # def optimize(self,
    #              init_points,
    #              n_iter,
    #              xi,
    #              weight_bound_min,
    #              weight_bound_max,
    #              use_these_thresholds=None,
    #              use_these_weights=None):
    #     timestamp = UtilityFuncs.get_timestamp()
    #     for dataset in self.listOfRoutingData:













        self.listOfResults = []
        train_indices = self.routingData.trainingIndices
        test_indices = self.routingData.testIndices
        self.fixedWeights = None
        self.fixedThresholds = None
        # Learn the standard information gain based accuracies
        train_ig_accuracy = InformationGainRoutingAccuracyCalculator.calculate(network=self.network,
                                                                               routing_data=self.routingData,
                                                                               indices=train_indices)
        test_ig_accuracy = InformationGainRoutingAccuracyCalculator.calculate(network=self.network,
                                                                              routing_data=self.routingData,
                                                                              indices=test_indices)
        print("train_ig_accuracy={0}".format(train_ig_accuracy))
        print("test_ig_accuracy={0}".format(test_ig_accuracy))
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # Three modes:
        # 1-Routing Weights fixed, Thresholds are optimized
        # 2-Routing Weights are optimized, Thresholds are fixed
        # 3-Routing Weights and Thresholds are optimized

        threshold_bounds = self.calculate_threshold_bounds()
        weight_bounds = self.calculate_weight_bounds(min_boundary=weight_bound_min, max_boundary=weight_bound_max)
        all_bounds = {}
        # Optimize both
        if use_these_thresholds is None and use_these_weights is None:
            assert weight_bound_min is not None and weight_bound_max is not None
            all_bounds.update(threshold_bounds)
            all_bounds.update(weight_bounds)
        # Weights fixed; optimize thresholds
        elif use_these_thresholds is None and use_these_weights is not None:
            all_bounds.update(threshold_bounds)
            self.fixedWeights = use_these_weights
        # Thresholds fixed; optimize weights
        elif use_these_thresholds is not None and use_these_weights is None:
            all_bounds.update(weight_bounds)
            self.fixedThresholds = use_these_thresholds
        # Else; if both a threshold and weight array has been provided; don't optimize; just return the result.
        else:
            self.fixedThresholds = use_these_thresholds
            self.fixedWeights = use_these_weights
            results = self.get_thresholding_results_for_args({})
            return results

        # Actual optimization part
        optimizer = BayesianOptimization(
            f=self.loss_function,
            pbounds=all_bounds,
        )
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
            acq="ei",
            xi=xi
        )
        self.write_to_db(xi=xi, timestamp=timestamp)
        print("X")