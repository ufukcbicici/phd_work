import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs


class DqnWithRegression:
    invalid_action_penalty = -1.0
    valid_prediction_reward = 1.0
    invalid_prediction_penalty = 0.0
    INCLUDE_IG_IN_REWARD_CALCULATIONS = True

    CONV_FEATURES = [64]
    FILTER_SIZES = [1]
    STRIDES = [1]
    HIDDEN_LAYERS = [128, 64]
    MAX_POOL = [None]

    def __init__(self, routing_dataset, network, network_name, run_id, used_feature_names, q_learning_func,
                 lambda_mac_cost, max_experience_count=100000):
        self.routingDataset = routing_dataset
        self.network = network
        self.networkName = network_name
        self.runId = run_id
        self.usedFeatureNames = used_feature_names
        self.qLearningFunc = q_learning_func
        self.lambdaMacCost = lambda_mac_cost
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        self.innerNodes = sorted(self.innerNodes, key=lambda node: node.index)
        self.leafNodes = sorted(self.leafNodes, key=lambda node: node.index)
        self.leafIndices = {node.index: idx for idx, node in enumerate(self.leafNodes)}
        # Data containers
        self.maxLikelihoodPaths = None
        self.stateFeatures = {}
        self.posteriorTensors = None
        self.actionSpaces = []
        self.networkActivationCosts = None
        self.networkActivationCostsDict = None
        self.baseEvaluationCost = 0.0
        self.reachabilityMatrices = None
        self.rewardTensors = None
        # Init data structures
        self.get_max_likelihood_paths()
        self.prepare_state_features()
        # self.prepare_posterior_tensors()
        # self.build_action_spaces()
        # self.get_evaluation_costs()
        # self.get_reachability_matrices()
        # self.calculate_reward_tensors()
        # # Neural network components
        # self.experienceReplayTable = None
        # self.maxExpCount = max_experience_count
        # self.stateInput = None
        # self.qFunction = None
        # self.selectedQs = None
        # self.stateCount = tf.placeholder(dtype=tf.int32, name="stateCount")
        # self.stateRange = tf.range(0, self.stateCount, 1)
        # self.actionSelection = None
        # self.selectionMatrix = None
        # self.rewardVector = None
        # self.lossVector = None
        # self.lossValue = None
        # self.optimizer = None
        # self.totalLoss = None
        # self.globalStep = tf.Variable(0, name='global_step', trainable=False)
        # self.l2LambdaTf = tf.placeholder(dtype=tf.float32, name="l2LambdaTf")
        # self.l2Loss = None
        # self.build_q_function()
        # self.session = tf.Session()
        # # The following is for testing, can comment out later.
        # # self.test_likelihood_consistency()
        # print("X")

    # OK
    def get_max_trajectory_length(self) -> int:
        return int(self.network.depth - 1)

    # OK
    def get_max_likelihood_paths(self):
        branch_probs = self.routingDataset.get_dict("branch_probs")
        sample_sizes = list(set([arr.shape[0] for arr in branch_probs.values()]))
        assert len(sample_sizes) == 1
        sample_size = sample_sizes[0]
        max_likelihood_paths = []
        for idx in range(sample_size):
            curr_node = self.network.topologicalSortedNodes[0]
            route = []
            while True:
                route.append(curr_node.index)
                if curr_node.isLeaf:
                    break
                routing_distribution = branch_probs[curr_node.index][idx]
                arg_max_child_index = np.argmax(routing_distribution)
                child_nodes = self.network.dagObject.children(node=curr_node)
                child_nodes = sorted(child_nodes, key=lambda c_node: c_node.index)
                curr_node = child_nodes[arg_max_child_index]
            max_likelihood_paths.append(np.array(route))
        self.maxLikelihoodPaths = np.stack(max_likelihood_paths, axis=0)

    # OK
    def prepare_state_features(self):
        # if self.policyNetworkFunc == "mlp":
        #     super().prepare_state_features(data=data)
        # elif self.policyNetworkFunc == "cnn":
        root_node = [node for node in self.network.topologicalSortedNodes if node.isRoot]
        assert len(root_node) == 1
        for node in self.innerNodes:
            # array_list = [data.get_dict(feature_name)[node.index] for feature_name in self.networkFeatureNames]
            array_list = []
            for feature_name in self.usedFeatureNames:
                feature_arr = self.routingDataset.get_dict(feature_name)[node.index]
                if self.qLearningFunc == "mlp":
                    if len(feature_arr.shape) > 2:
                        shape_as_list = list(feature_arr.shape)
                        mean_axes = tuple([i for i in range(1, len(shape_as_list) - 1, 1)])
                        feature_arr = np.mean(feature_arr, axis=mean_axes)
                        assert len(feature_arr.shape) == 2
                elif self.qLearningFunc == "cnn":
                    assert len(feature_arr.shape) == 4
                array_list.append(feature_arr)
            feature_vectors = np.concatenate(array_list, axis=-1)
            self.stateFeatures[node.index] = feature_vectors




        # for iteration in self.routingDataset.iterations:
        #     features_dict = {}
        #     for node in self.innerNodes:
        #         # array_list = [data.get_dict(feature_name)[node.index] for feature_name in self.networkFeatureNames]
        #         array_list = []
        #         for feature_name in self.usedFeatureNames:
        #             feature_arr = self.routingDataset.dictOfDatasets[iteration].get_dict(feature_name)[node.index]
        #             if self.qLearningFunc == "mlp":
        #                 if len(feature_arr.shape) > 2:
        #                     shape_as_list = list(feature_arr.shape)
        #                     mean_axes = tuple([i for i in range(1, len(shape_as_list) - 1, 1)])
        #                     feature_arr = np.mean(feature_arr, axis=mean_axes)
        #                     assert len(feature_arr.shape) == 2
        #             elif self.qLearningFunc == "cnn":
        #                 assert len(feature_arr.shape) == 4
        #             array_list.append(feature_arr)
        #         feature_vectors = np.concatenate(array_list, axis=-1)
        #         features_dict[node.index] = feature_vectors
        #     self.stateFeatures[iteration] = features_dict
        #


