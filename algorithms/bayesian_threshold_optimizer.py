import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization

from algorithms.information_gain_routing_accuracy_calculator import InformationGainRoutingAccuracyCalculator
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer import \
    DirectThresholdOptimizer
from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer_entropy import \
    DirectThresholdOptimizerEntropy
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs


class BayesianThresholdOptimizer:
    def __init__(self,
                 network, routing_data, session, seed, temperatures_dict,
                 threshold_kind, mixing_lambda, run_id):
        self.network = network
        self.routingData = routing_data
        self.session = session
        self.seed = seed
        self.runId = run_id
        self.temperaturesDict = temperatures_dict
        self.thresholdKind = threshold_kind
        self.mixingLambda = mixing_lambda
        if self.thresholdKind == "entropy":
            self.thresholdOptimizer = DirectThresholdOptimizerEntropy(
                network=self.network, routing_data=self.routingData,
                seed=self.seed, use_parametric_weights=True)
        else:
            self.thresholdOptimizer = DirectThresholdOptimizer(
                network=self.network, routing_data=self.routingData,
                seed=self.seed,
                use_parametric_weights=True)
        self.thresholdOptimizer.build_network()
        self.fixedThresholds = None
        self.fixedWeights = None
        self.listOfResults = []

    def get_random_thresholds(self):
        thresholds_dict = {}
        for node in self.network.innerNodes:
            child_nodes = self.network.dagObject.children(node)
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
        return thresholds_dict

    def calculate_threshold_bounds(self):
        pbounds = {}
        for node in self.network.innerNodes:
            child_nodes = self.network.dagObject.children(node)
            child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
            if self.thresholdKind == "probability":
                max_bound = 1.0 / len(child_nodes)
                for c_nd in child_nodes:
                    pbounds["t_{0}_{1}".format(node.index, c_nd.index)] = (0.0, max_bound)
            elif self.thresholdKind == "entropy":
                max_bound = -np.log(1.0 / len(child_nodes))
                pbounds["t_{0}".format(node.index)] = (0.0, max_bound)
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
        for node in self.network.leafNodes:
            pbounds["leaf_{0}".format(node.index)] = (min_boundary, max_boundary)
        return pbounds

    def decode_threshold_parameters(self, args_dict):
        thrs_dict = {}
        for node in self.network.innerNodes:
            child_nodes = self.network.dagObject.children(node)
            child_nodes = sorted(child_nodes, key=lambda nd: nd.index)
            if self.thresholdKind == "probability":
                thrs_arr = np.array([args_dict["t_{0}_{1}".format(node.index, c_nd.index)]
                                     for c_nd in child_nodes])
                thrs_dict[node.index] = thrs_arr[np.newaxis, :]
            elif self.thresholdKind == "entropy":
                thrs_dict[node.index] = args_dict["t_{0}".format(node.index)]
            else:
                raise NotImplementedError()
        return thrs_dict

    def decode_weight_parameters(self, args_dict):
        weight_list = []
        for node in self.network.leafNodes:
            weight = args_dict["leaf_{0}".format(node.index)]
            weight_list.append(weight)
        weight_list = np.array(weight_list)[np.newaxis, :]
        return weight_list

    def get_thresholding_results(self,
                                 indices,
                                 thresholds_dict,
                                 routing_weights_array,
                                 temperatures_dict,
                                 mixing_lambda):
        # Convert to probabilities
        weights_as_probabilities = np.exp(routing_weights_array) * np.reciprocal(np.sum(np.exp(routing_weights_array)))
        weights_array = np.repeat(weights_as_probabilities, repeats=indices.shape[0], axis=0)
        optimizer_results = self.thresholdOptimizer.run_threshold_calculator(
            sess=self.session,
            routing_data=self.routingData,
            indices=indices,
            mixing_lambda=mixing_lambda,
            temperatures_dict=temperatures_dict,
            thresholds_dict=thresholds_dict,
            routing_weights_array=weights_array)
        correctness_vector = optimizer_results["correctnessVector"]
        activation_costs_vector = optimizer_results["activationCostsArr"]
        selection_tuples = optimizer_results["selectionTuples"]
        selection_weights = optimizer_results["selectionWeights"]
        posteriors = optimizer_results["posteriorsTensor"]
        # Calculate required metrics
        score_vector = mixing_lambda * correctness_vector - (1.0 - mixing_lambda) * activation_costs_vector
        final_score = np.mean(score_vector)
        final_accuracy = np.mean(correctness_vector)
        final_activation_cost = np.mean(activation_costs_vector)
        # Results
        results_dict = {"final_score": final_score,
                        "score_vector": score_vector,
                        "final_accuracy": final_accuracy,
                        "correctness_vector": correctness_vector,
                        "final_activation_cost": final_activation_cost,
                        "activation_costs_vector": activation_costs_vector,
                        "selection_tuples": selection_tuples,
                        "selection_weights": selection_weights}
        return results_dict

    def get_thresholding_results_for_args(self, kwargs):
        # Convert Bayesian Optimization space sample into usable thresholds
        thresholds = self.decode_threshold_parameters(kwargs) if self.fixedThresholds is None else self.fixedThresholds
        weights = self.decode_weight_parameters(kwargs) if self.fixedWeights is None else self.fixedWeights
        train_indices = self.routingData.trainingIndices
        test_indices = self.routingData.testIndices
        sample_indices = {"train": train_indices, "test": test_indices}
        results_dict = {}
        for data_type in ["train", "test"]:
            results = self.get_thresholding_results(
                indices=sample_indices[data_type],
                thresholds_dict=thresholds,
                routing_weights_array=weights,
                temperatures_dict=self.temperaturesDict,
                mixing_lambda=self.mixingLambda)
            results_dict[data_type] = results
        print("Train Accuracy: {0} Train Computation Load:{1} Train Score:{2}".format(
            results_dict["train"]["final_accuracy"],
            results_dict["train"]["final_activation_cost"], results_dict["train"]["final_score"]))
        print("Test Accuracy: {0} Test Computation Load:{1} Test Score:{2}".format(
            results_dict["test"]["final_accuracy"],
            results_dict["test"]["final_activation_cost"], results_dict["test"]["final_score"]))
        return results_dict

    def loss_function(self, **kwargs):
        results_dict = self.get_thresholding_results_for_args(kwargs)
        self.listOfResults.append(results_dict)
        return results_dict["train"]["final_score"]

    def write_to_db(self, xi, timestamp):
        db_rows = []
        # result = (iteration_id, new_score, new_accuracy, new_computation_overload)
        for result in self.listOfResults:
            val_score = result["train"]["final_score"]
            val_accuracy = result["train"]["final_accuracy"]
            val_overload = result["train"]["final_activation_cost"]
            test_score = result["test"]["final_score"]
            test_accuracy = result["test"]["final_accuracy"]
            test_overload = result["test"]["final_activation_cost"]
            db_rows.append((self.runId,
                            self.network.networkName,
                            -1,
                            self.mixingLambda,
                            0,
                            val_score, val_accuracy, val_overload,
                            test_score, test_accuracy, test_overload,
                            "Bayesian Optimization", xi, timestamp))
        DbLogger.write_into_table(rows=db_rows, table=DbLogger.threshold_optimization, col_count=-1)

    def optimize(self,
                 init_points,
                 n_iter,
                 xi,
                 weight_bound_min,
                 weight_bound_max,
                 use_these_thresholds=None,
                 use_these_weights=None):
        timestamp = UtilityFuncs.get_timestamp()
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
