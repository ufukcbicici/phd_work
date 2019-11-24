import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class RoutingWeightCalculator:
    def __init__(self, network, validation_routing_matrix, test_routing_matrix, validation_data, test_data):
        self.network = network
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        self.innerNodes = sorted(self.innerNodes, key=lambda node: node.index)
        self.leafNodes = sorted(self.leafNodes, key=lambda node: node.index)
        self.validationRoutingMatrix = validation_routing_matrix
        self.testRoutingMatrix = test_routing_matrix
        self.validationData = validation_data
        self.testData = test_data

    def get_visited_parent_nodes(self, routing_vector):
        assert len(routing_vector) == len(self.leafNodes)
        parent_node_set = set()
        for idx, is_open in enumerate(routing_vector):
            if is_open == 0:
                continue
            leaf_node = self.leafNodes[idx]
            leaf_ancestors = self.network.dagObject.ancestors(node=leaf_node)
            for ancestor_node in leaf_ancestors:
                parent_node_set.add(ancestor_node.index)
        parent_node_routing_vector = np.array([node.index in parent_node_set for node in self.innerNodes],
                                              dtype=np.int32)
        return parent_node_routing_vector

    def run(self):
        # Create Feature Sets from Activation and Posterior Vectors
        features_dict = {}
        targets_dict = {}
        for routing_matrix, data, data_type in zip([self.validationRoutingMatrix, self.testRoutingMatrix],
                                        [self.validationData, self.testData],
                                        ["validation", "test"]):
            posterior_features = []
            routing_activation_features = []
            targets = []
            correct_count_single_path = 0
            correct_count_simple_avg = 0
            correct_count_least_squares = 0
            single_path_indices = np.nonzero(np.sum(routing_matrix, axis=1) == 1)[0]
            multi_path_indices = np.nonzero(np.sum(routing_matrix, axis=1) > 1)[0]
            posteriors_per_leaf = sorted([(k, v) for k, v in data.posteriorProbs.items()], key=lambda tpl: tpl[0])
            posteriors_tensor = np.stack([tpl[1] for tpl in posteriors_per_leaf], axis=2)
            activations_per_inner_node = sorted([(k, v) for k, v in data.activations.items()], key=lambda tpl: tpl[0])
            activations_tensor = np.stack([tpl[1] for tpl in activations_per_inner_node], axis=2)
            for idx in range(routing_matrix.shape[0]):
                routing_vector = routing_matrix[idx]
                posterior_matrix = posteriors_tensor[idx, :, :]
                activations_matrix = activations_tensor[idx, :, :]
                inner_nodes_routing_vector = self.get_visited_parent_nodes(routing_vector=routing_vector)
                label_vector = np.zeros_like(posterior_matrix[:, 0])
                true_label = data.labelList[idx]
                label_vector[true_label] = 1.0
                # Simple Averaging
                leaf_count = np.sum(routing_vector)
                assert len(self.leafNodes) >= leaf_count >= 1
                sparse_posteriors_matrix = posterior_matrix * np.expand_dims(routing_vector, axis=0)
                sparse_activations_matrix = activations_matrix * np.expand_dims(inner_nodes_routing_vector, axis=0)
                avg_posteriors = np.sum(sparse_posteriors_matrix, axis=1) / leaf_count
                simple_avg_predicted_label = np.argmax(avg_posteriors)
                if leaf_count == 1:
                    correct_count_single_path += int(simple_avg_predicted_label == true_label)
                else:
                    correct_count_simple_avg += int(simple_avg_predicted_label == true_label)
                # Least Square Weights
                if leaf_count == 1:
                    continue
                res = np.linalg.lstsq(sparse_posteriors_matrix, label_vector, rcond=None)
                alpha_lst_squares = res[0]
                posterior_feature_vector = \
                    np.reshape(sparse_posteriors_matrix.T,
                               newshape=(sparse_posteriors_matrix.T.shape[0] * sparse_posteriors_matrix.T.shape[1], ))
                activation_feature_vector = \
                    np.reshape(sparse_activations_matrix.T,
                               newshape=(sparse_activations_matrix.T.shape[0] * sparse_activations_matrix.T.shape[1], ))
                target_vector = alpha_lst_squares
                posterior_features.append(posterior_feature_vector)
                routing_activation_features.append(activation_feature_vector)
                targets.append(target_vector)
                lst_squares_posterior = posterior_matrix @ alpha_lst_squares
                lst_squares_predicted_label = np.argmax(lst_squares_posterior)
                correct_count_least_squares += int(lst_squares_predicted_label == true_label)
            multi_path_accuracy_simple_avg = (correct_count_single_path + correct_count_simple_avg) \
                                             / routing_matrix.shape[0]
            multi_path_accuracy_lst_squares = (correct_count_single_path + correct_count_least_squares) \
                                              / routing_matrix.shape[0]
            print("Simple Mean Avg Accuracy:{0}".format(multi_path_accuracy_simple_avg))
            print("Least Squares Avg Accuracy:{0}".format(multi_path_accuracy_lst_squares))
            posterior_features_matrix = np.stack(posterior_features, axis=0)
            activation_features_matrix = np.stack(routing_activation_features, axis=0)
            weights_target_matrix = np.stack(targets, axis=0)
            features_dict[data_type] = [posterior_features_matrix, activation_features_matrix]
            targets_dict[data_type] = [weights_target_matrix]

        # Modelling
        x_train = features_dict["validation"][0]
        y_train = targets_dict["validation"]

        # RDF
        feature_dim = x_train.shape[1]
        pca = PCA()
        rdf = RandomForestRegressor(criterion="mse")
        pipe = Pipeline(steps=[('pca', pca), ('rdf', rdf)])
        step = max(1, int((feature_dim - 5) / 10))
        # Hyperparameter grid
        pca__n_components = [d for d in range(5, feature_dim, step)]
        pca__n_components.append(feature_dim)
        param_grid = {
            'pca__n_components': pca__n_components,
            'rdf__n_estimators': [100],
            'rdf__max_depth': [5, 10, 15, 20, 25, 30],
            'rdf__bootstrap': [False, True],
            'rdf__oob_score': [False, True],
            'rdf__min_samples_leaf': [1, 2, 3, 4, 5, 10]
        }
        grid_search = GridSearchCV(pipe, param_grid, iid=False, cv=5, n_jobs=8, refit=True, verbose=0)
        grid_search.fit(X=x_train, y=y_train)
        best_model = grid_search.best_estimator_
        print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
        print(grid_search.best_params_)
        print("X")

