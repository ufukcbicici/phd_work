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
                 list_of_networks, list_of_routing_data,
                 session, seed, temperatures_dict, threshold_kind, mixing_lambda, run_id):
        self.listOfNetworks = list_of_networks
        self.listOfRoutingData = list_of_routing_data
        assert len(list_of_networks) == len(list_of_routing_data)
        self.ensembleSize = len(self.listOfNetworks)
        self.session = session
        self.seed = seed
        self.runId = run_id
        self.temperaturesDict = temperatures_dict
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



        # train_indices = self.routingData.trainingIndices
        # test_indices = self.routingData.testIndices
        # sample_indices = {"train": train_indices, "test": test_indices}
        # results_dict = {}
        # for data_type in ["train", "test"]:
        #     results = self.get_thresholding_results(
        #         indices=sample_indices[data_type],
        #         thresholds_dict=thresholds,
        #         routing_weights_array=weights,
        #         temperatures_dict=self.temperaturesDict,
        #         mixing_lambda=self.mixingLambda)
        #     results_dict[data_type] = results
        # print("Train Accuracy: {0} Train Computation Load:{1} Train Score:{2}".format(
        #     results_dict["train"]["final_accuracy"],
        #     results_dict["train"]["final_activation_cost"], results_dict["train"]["final_score"]))
        # print("Test Accuracy: {0} Test Computation Load:{1} Test Score:{2}".format(
        #     results_dict["test"]["final_accuracy"],
        #     results_dict["test"]["final_activation_cost"], results_dict["test"]["final_score"]))
        # return results_dict
