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
        single_path_correct_counts_dict = {}
        multi_path_indices_dict = {}
        posterior_tensors_dict = {}
        activation_tensors_dict = {}
        data_objects_dict = {"validation": self.validationData, "test": self.testData}
        routing_matrices_dict = {"validation": self.validationRoutingMatrix, "test": self.testRoutingMatrix}
        for routing_matrix, data, data_type in zip([self.validationRoutingMatrix, self.testRoutingMatrix],
                                                   [self.validationData, self.testData],
                                                   ["validation", "test"]):
            posterior_features = []
            routing_activation_features = []
            targets = []
            correct_count_single_path = 0
            correct_count_simple_avg = 0
            correct_count_least_squares = 0
            single_path_indices_ = np.nonzero(np.sum(routing_matrix, axis=1) == 1)[0]
            multi_path_indices_ = np.nonzero(np.sum(routing_matrix, axis=1) > 1)[0]
            multi_path_indices_dict[data_type] = multi_path_indices_
            posteriors_per_leaf = sorted([(k, v) for k, v in data.posteriorProbs.items()], key=lambda tpl: tpl[0])
            posteriors_tensor = np.stack([tpl[1] for tpl in posteriors_per_leaf], axis=2)
            posterior_tensors_dict[data_type] = posteriors_tensor
            activations_per_inner_node = sorted([(k, v) for k, v in data.activations.items()], key=lambda tpl: tpl[0])
            activations_tensor = np.stack([tpl[1] for tpl in activations_per_inner_node], axis=2)
            activation_tensors_dict[data_type] = activations_tensor
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
                assert idx == multi_path_indices_[len(targets)]
                res = np.linalg.lstsq(sparse_posteriors_matrix, label_vector, rcond=None)
                alpha_lst_squares = res[0]
                posterior_feature_vector = \
                    np.reshape(sparse_posteriors_matrix.T,
                               newshape=(sparse_posteriors_matrix.T.shape[0] * sparse_posteriors_matrix.T.shape[1],))
                activation_feature_vector = \
                    np.reshape(sparse_activations_matrix.T,
                               newshape=(sparse_activations_matrix.T.shape[0] * sparse_activations_matrix.T.shape[1],))
                target_vector = alpha_lst_squares
                posterior_features.append(posterior_feature_vector)
                routing_activation_features.append(activation_feature_vector)
                targets.append(target_vector)
                lst_squares_posterior = sparse_posteriors_matrix @ alpha_lst_squares
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
            single_path_correct_counts_dict[data_type] = correct_count_single_path

        # Modelling
        def get_feature_vectors(data_type_):
            posterior_vecs = features_dict[data_type_][0]
            activation_vecs = features_dict[data_type_][1]
            x_train = np.concatenate([posterior_vecs, activation_vecs], axis=1)
            return x_train

        x_train = get_feature_vectors("validation")
        y_train = targets_dict["validation"][0]

        # RDF
        feature_dim = x_train.shape[1]
        pca = PCA()
        rdf = RandomForestRegressor(criterion="mse")
        pipe = Pipeline(steps=[('pca', pca), ('rdf', rdf)])
        step = max(1, int((feature_dim - 5) / 50))
        # Hyperparameter grid
        # pca__n_components = [d for d in range(5, feature_dim, step)]
        # pca__n_components.append(feature_dim)
        pca__n_components = [50]
        param_grid = {
            'pca__n_components': pca__n_components,
            'rdf__n_estimators': [1000],
            'rdf__max_depth': [5, 10, 15, 20, 25, 30],
            'rdf__bootstrap': [False, True],
            'rdf__min_samples_leaf': [1, 2, 3, 4, 5, 10]
        }
        grid_search = GridSearchCV(pipe, param_grid, iid=False, cv=5, n_jobs=8, refit=True, verbose=10)
        grid_search.fit(X=x_train, y=y_train)
        best_model = grid_search.best_estimator_
        print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
        print(grid_search.best_params_)

        # Predicting with the regressed weights
        for data_type in ["validation", "test"]:
            correct_count = 0
            real_least_squares_correct_count = 0
            posterior_features = features_dict[data_type][0]
            activation_features = features_dict[data_type][1]
            lst_weights = targets_dict[data_type][0]
            feature_vectors = get_feature_vectors(data_type)
            weights_predicted = best_model.predict(X=feature_vectors)
            routing_matrix = routing_matrices_dict[data_type]
            posteriors_tensor = posterior_tensors_dict[data_type]
            multi_path_indices_ = multi_path_indices_dict[data_type]
            data_obj = data_objects_dict[data_type]
            for idx, multi_path_idx in enumerate(multi_path_indices_):
                true_label = data_obj.labelList[multi_path_idx]
                routing_vector = routing_matrix[multi_path_idx]
                posterior_matrix = posteriors_tensor[multi_path_idx, :, :]
                weights = weights_predicted[idx, :]
                lst_weights_ = lst_weights[idx, :]
                sparse_posteriors_matrix = posterior_matrix * np.expand_dims(routing_vector, axis=0)
                lst_squares_posterior_regressed = sparse_posteriors_matrix @ weights
                lst_squares_regressor_predicted_label = np.argmax(lst_squares_posterior_regressed)
                correct_count += int(lst_squares_regressor_predicted_label == true_label)
                lst_squares_posterior = sparse_posteriors_matrix @ lst_weights_
                lst_squares_predicted_label = np.argmax(lst_squares_posterior)
                real_least_squares_correct_count += int(lst_squares_predicted_label == true_label)
            multi_path_accuracy_lst_squares_regression = (single_path_correct_counts_dict[data_type] + correct_count) \
                                                         / routing_matrix.shape[0]
            multi_path_accuracy_lst_squares = (single_path_correct_counts_dict[
                                                   data_type] + real_least_squares_correct_count) \
                                              / routing_matrix.shape[0]
            print("Regression Result DataType:{0} Accuracy:{1}".format(data_type,
                                                                       multi_path_accuracy_lst_squares_regression))
            print("Actual Least Squares Result DataType:{0} Accuracy:{1}".format(data_type,
                                                                                 multi_path_accuracy_lst_squares))
        print("X")

    # def model_for_separate_leaves(self):
