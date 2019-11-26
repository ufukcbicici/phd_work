import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from algorithms.threshold_optimization_algorithms.routing_weight_calculator import RoutingWeightCalculator
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.global_params import GlobalConstants


class RoutingWeightsFinderWithLeastSquares(RoutingWeightCalculator):
    def __init__(self, network, validation_routing_matrix, test_routing_matrix, validation_data, test_data,
                 min_cluster_size):
        super().__init__(network, validation_routing_matrix, test_routing_matrix, validation_data, test_data)
        self.routingCombinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(self.leafNodes))
        self.routingCombinations = [np.array(route_vec) for route_vec in self.routingCombinations]
        self.routingCombinationClustersDict = {}
        self.routingCombinationClusterWeightsDict = {}
        self.minClusterSize = min_cluster_size
        formatted_data_validation = RoutingWeightCalculator.format_routing_data(routing_data=self.validationData)
        formatted_data_test = RoutingWeightCalculator.format_routing_data(routing_data=self.testData)
        self.featuresDict = {"validation": formatted_data_validation, "test": formatted_data_test}

    def get_feature_vectors(self, routing_vector, **kwargs):
        posterior_tensor = kwargs["posterior_probs"]
        activations_tensor = kwargs["activations"]
        ancestral_nodes_routing_vector = self.get_visited_parent_nodes(routing_vector)
        sparse_posteriors_tensor = posterior_tensor * np.expand_dims(np.expand_dims(routing_vector, axis=0), axis=0)
        sparse_activations_tensor = activations_tensor * np.expand_dims(
            np.expand_dims(ancestral_nodes_routing_vector, axis=0), axis=0)
        posterior_features = np.reshape(sparse_posteriors_tensor, newshape=(
            sparse_posteriors_tensor.shape[0], sparse_posteriors_tensor.shape[1] * sparse_posteriors_tensor.shape[2]))
        activation_features = np.reshape(sparse_activations_tensor, newshape=(
            sparse_activations_tensor.shape[0],
            sparse_activations_tensor.shape[1] * sparse_activations_tensor.shape[2]))
        # Only activations
        X = activation_features
        return X, sparse_posteriors_tensor

    def run(self):
        # Create Feature Sets from Activation and Posterior Vectors
        # features_dict = {}
        # targets_dict = {}
        # single_path_correct_counts_dict = {}
        # multi_path_indices_dict = {}
        # posterior_tensors_dict = {}
        # activation_tensors_dict = {}
        # data_objects_dict = {"validation": self.validationData, "test": self.testData}
        # routing_matrices_dict = {"validation": self.validationRoutingMatrix, "test": self.testRoutingMatrix}
        for route_vector in self.routingCombinations:
            if np.sum(route_vector) <= 1:
                continue
            element_wise_compliance = self.validationRoutingMatrix == route_vector
            valid_samples_indicator_vector = np.all(element_wise_compliance, axis=1)
            posteriors_tensor = np.copy(
                self.featuresDict["validation"]["posterior_probs"][valid_samples_indicator_vector, :])
            activations_tensor = np.copy(
                self.featuresDict["validation"]["activations"][valid_samples_indicator_vector, :])
            X, sparse_posteriors_tensor = self.get_feature_vectors(routing_vector=route_vector,
                                                                   posterior_probs=posteriors_tensor,
                                                                   activations=activations_tensor)
            y = self.validationData.labelList[valid_samples_indicator_vector]
            cluster_count = max(int(X.shape[0] / self.minClusterSize), 1)
            # while True:
            #     kmeans = KMeans(n_clusters=cluster_count, random_state=0)
            #     kmeans.fit(X)
            #     cluster_labels = kmeans.predict(X)
            #     counter = Counter(cluster_labels)
            #     least_freq = counter.most_common()[-1][1]
            #     if least_freq >= self.minClusterSize:
            #         break
            #     cluster_count -= 1
            #     if cluster_count == 1:
            #         break
            kmeans = KMeans(n_clusters=cluster_count, random_state=0)
            kmeans.fit(X)
            cluster_labels = kmeans.predict(X)
            route_vector_as_tuple = tuple(route_vector.tolist())
            self.routingCombinationClustersDict[route_vector_as_tuple] = kmeans
            self.routingCombinationClusterWeightsDict[route_vector_as_tuple] = []
            # Least Squares fitting
            for cluster_id in range(cluster_count):
                i_vec = cluster_labels == cluster_id
                sparse_posteriors_tensor_cluster = sparse_posteriors_tensor[i_vec, :]
                y_cluster = y[i_vec]
                dim = sparse_posteriors_tensor.shape[1]
                cluster_size = y_cluster.shape[0]
                assert y_cluster.shape[0] == sparse_posteriors_tensor_cluster.shape[0]
                b = np.zeros((cluster_size * dim, ))
                label_positions = (dim * np.arange(0, cluster_size)) + y_cluster
                b[label_positions] = 1.0
                arr_list = [sparse_posteriors_tensor_cluster[idx, :] for idx in range(cluster_size)]
                A = np.concatenate(arr_list, axis=0)
                res = np.linalg.lstsq(A, b, rcond=None)
                alpha_weights = res[0]
                self.routingCombinationClusterWeightsDict[route_vector_as_tuple].append(alpha_weights)
        #         print("X")
        # print("X")
        self.estimate_accuracy(data_type="validation", routing_matrix=self.validationRoutingMatrix)
        self.estimate_accuracy(data_type="test", routing_matrix=self.testRoutingMatrix)

    def estimate_accuracy(self, data_type, routing_matrix):
        correct_count = 0
        data = self.featuresDict[data_type]
        for idx in range(routing_matrix.shape[0]):
            routing_vector = routing_matrix[idx, :]
            route_vector_as_tuple = tuple(routing_vector.tolist())
            posteriors = np.expand_dims(data["posterior_probs"][idx, :], axis=0)
            activations = np.expand_dims(data["activations"][idx, :], axis=0)
            x, sparse_posteriors_tensor = self.get_feature_vectors(routing_vector=routing_vector,
                                                                   posterior_probs=posteriors,
                                                                   activations=activations)
            sparse_posteriors_tensor = sparse_posteriors_tensor[0, :]
            if np.sum(routing_vector) == 1:
                weights = routing_vector
            else:
                kmeans = self.routingCombinationClustersDict[route_vector_as_tuple]
                cluster_id = kmeans.predict(x)
                weights = self.routingCombinationClusterWeightsDict[cluster_id]
            least_squares_posterior = sparse_posteriors_tensor @ weights
            predicted_label = np.argmax(least_squares_posterior)
            true_label = data.labelList[idx]
            correct_count += int(predicted_label == true_label)
        accuracy = correct_count / routing_matrix.shape[0]
        print("{0} accuracy is:{1}".format(data_type, accuracy))
