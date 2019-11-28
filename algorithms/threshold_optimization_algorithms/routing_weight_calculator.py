import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from simple_tf.global_params import GlobalConstants


class RoutingWeightCalculator:
    def __init__(self, network, validation_routing_matrix, test_routing_matrix, validation_data, test_data):
        self.network = network
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        self.innerNodes = sorted(self.innerNodes, key=lambda node: node.index)
        self.leafNodes = sorted(self.leafNodes, key=lambda node: node.index)
        self.validationRoutingMatrix = validation_routing_matrix
        self.testRoutingMatrix = test_routing_matrix
        self.validationInnerRoutingMatrix = np.apply_along_axis(self.get_visited_parent_nodes, axis=1,
                                                                arr=self.validationRoutingMatrix)
        self.testInnerRoutingMatrix = np.apply_along_axis(self.get_visited_parent_nodes, axis=1,
                                                          arr=self.testRoutingMatrix)
        self.validationData = validation_data
        self.testData = test_data
        self.validationSparsePosteriors = None
        self.testSparsePosteriors = None

    def build_data_sets(self, selected_features):
        inner_features = set(GlobalConstants.INNER_NODE_OUTPUTS_TO_COLLECT)
        leaf_features = set(GlobalConstants.LEAF_NODE_OUTPUTS_TO_COLLECT)
        X_list = []
        Y_list = []
        sparse_posteriors = []
        for inner_routing_matrix, leaf_routing_matrix, data in \
                zip([self.validationInnerRoutingMatrix, self.testInnerRoutingMatrix],
                    [self.validationRoutingMatrix, self.testRoutingMatrix],
                    [self.validationData, self.testData]):
            # Build feature matrix (X)
            feature_matrices = []
            for feature_name in selected_features:
                assert feature_name in inner_features or feature_name in leaf_features
                nodes = self.leafNodes if feature_name in leaf_features else self.innerNodes
                routing_matrix = leaf_routing_matrix if feature_name in leaf_features else inner_routing_matrix
                matrix_list = [data.get_dict(feature_name)[node.index] for node in nodes]
                if any([matrix is None for matrix in matrix_list]):
                    print("Skipping feature {0} since it is missing.".format(feature_name))
                    continue
                # Sparse - Concatenate
                # feature_matrix = np.concatenate(matrix_list, axis=1)
                # feature_dim = matrix_list[0].shape[1]
                # routing_matrix_rpt = np.repeat(routing_matrix, axis=1, repeats=feature_dim)
                # assert feature_matrix.shape == routing_matrix_rpt.shape
                # sparse_feature_matrix = routing_matrix_rpt * feature_matrix
                # feature_matrices.append(sparse_feature_matrix)

                # Mean
                feature_matrix = np.stack(matrix_list, axis=2)
                sparse_feature_matrix = feature_matrix * np.expand_dims(routing_matrix, axis=1)
                feature_matrix = np.mean(sparse_feature_matrix, axis=2)
                feature_matrices.append(feature_matrix)
            X = np.concatenate(feature_matrices, axis=1)
            X = np.concatenate([X, leaf_routing_matrix], axis=1)
            X_list.append(X)
            # Build target matrix (Y)
            label_list = data.labelList
            posteriors_list = [np.expand_dims(data.get_dict("posterior_probs")[node.index], axis=2)
                               for node in self.leafNodes]
            posteriors_tensor = np.concatenate(posteriors_list, axis=2)
            sparse_posteriors_tensor = posteriors_tensor * np.expand_dims(leaf_routing_matrix, axis=1)
            sparse_posteriors.append(sparse_posteriors_tensor)
            y_list = []
            for idx in range(leaf_routing_matrix.shape[0]):
                route_vector = leaf_routing_matrix[idx]
                posteriors_matrix = posteriors_tensor[idx, :]
                sparse_posterior_matrix = posteriors_matrix * np.expand_dims(route_vector, axis=0)
                assert np.array_equal(sparse_posterior_matrix, sparse_posteriors_tensor[idx, :])
                target_label = label_list[idx]
                A = sparse_posterior_matrix
                b = self.get_one_hot_label_vector(target_label, dim=A.shape[0])
                res = np.linalg.lstsq(A, b, rcond=None)
                y = res[0]
                y_list.append(y)
            Y = np.stack(y_list, axis=0)
            Y_list.append(Y)
        validation_X = X_list[0]
        validation_Y = Y_list[0]
        self.validationSparsePosteriors = sparse_posteriors[0]
        test_X = X_list[1]
        test_Y = Y_list[1]
        self.testSparsePosteriors = sparse_posteriors[1]
        return validation_X, validation_Y, test_X, test_Y

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

    @staticmethod
    def format_routing_data(routing_data):
        formatted_data_dict = {}
        # Posteriors
        posterior_probs_dict = routing_data.get_dict("posterior_probs")
        posteriors_per_leaf = sorted([(k, v) for k, v in posterior_probs_dict.items()], key=lambda tpl: tpl[0])
        posteriors_tensor = np.stack([tpl[1] for tpl in posteriors_per_leaf], axis=2)
        formatted_data_dict["posterior_probs"] = posteriors_tensor
        # Branching Activations
        activations_dict = routing_data.get_dict("activations")
        activations_per_inner_node = sorted([(k, v) for k, v in activations_dict.items()], key=lambda tpl: tpl[0])
        activations_tensor = np.stack([tpl[1] for tpl in activations_per_inner_node], axis=2)
        formatted_data_dict["activations"] = activations_tensor
        return formatted_data_dict

    @staticmethod
    def get_one_hot_label_vector(label, dim):
        label_vec = np.zeros((dim,))
        label_vec[label] = 1.0
        return label_vec

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
            posterior_probs_dict = data.get_dict("posterior_probs")
            activations_dict = data.get_dict("activations")
            posterior_features = []
            routing_activation_features = []
            targets = []
            correct_count_single_path = 0
            correct_count_simple_avg = 0
            correct_count_least_squares = 0
            single_path_indices_ = np.nonzero(np.sum(routing_matrix, axis=1) == 1)[0]
            multi_path_indices_ = np.nonzero(np.sum(routing_matrix, axis=1) > 1)[0]
            multi_path_indices_dict[data_type] = multi_path_indices_
            posteriors_per_leaf = sorted([(k, v) for k, v in posterior_probs_dict.items()], key=lambda tpl: tpl[0])
            posteriors_tensor = np.stack([tpl[1] for tpl in posteriors_per_leaf], axis=2)
            posterior_tensors_dict[data_type] = posteriors_tensor
            activations_per_inner_node = sorted([(k, v) for k, v in activations_dict.items()], key=lambda tpl: tpl[0])
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
            # x_ = np.concatenate([posterior_vecs, activation_vecs], axis=1)
            x_ = activation_vecs
            return x_

        def get_rdf_regressor():
            rdf = RandomForestRegressor(criterion="mse")
            params = {
                'rdf__n_estimators': [1000],
                'rdf__max_depth': [5, 10, 15, 20, 25, 30, 40, 50],
                'rdf__bootstrap': [False, True],
                'rdf__min_samples_leaf': [1]
            }
            return ("rdf", rdf), params

        def get_mlp_regressor():
            mlp = MLPRegressor()
            params = {
                'mlp__hidden_layer_sizes': [(200, 100)],
                'mlp__activation': ["relu", "tanh"],
                'mlp__solver': ["lbfgs"],
                'mlp__alpha': [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.002],
                'mlp__max_iter': [10000]
            }
            return ("mlp", mlp), params

        x_train = get_feature_vectors("validation")
        y_train = targets_dict["validation"][0]
        validation_routing_matrix = np.copy(
            routing_matrices_dict["validation"][multi_path_indices_dict["validation"], :])

        # Model with single Regressor multiple output
        # RDF
        # feature_dim = x_train.shape[1]
        # pca = PCA()
        # rdf = RandomForestRegressor(criterion="mse")
        # pipe = Pipeline(steps=[('pca', pca), ('rdf', rdf)])
        # step = max(1, int((feature_dim - 5) / 50))
        # # Hyperparameter grid
        # # pca__n_components = [d for d in range(5, feature_dim, step)]
        # # pca__n_components.append(feature_dim)
        # pca__n_components = [50]
        # param_grid = {
        #     'pca__n_components': pca__n_components,
        #     'rdf__n_estimators': [100],
        #     'rdf__max_depth': [5, 10, 15, 20, 25, 30],
        #     'rdf__bootstrap': [False, True],
        #     'rdf__min_samples_leaf': [1, 2, 3, 4, 5, 10]
        # }
        # grid_search = GridSearchCV(pipe, param_grid, iid=False, cv=5, n_jobs=8, refit=True, verbose=10)
        # grid_search.fit(X=x_train, y=y_train)
        # best_model = grid_search.best_estimator_
        # print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
        # print(grid_search.best_params_)
        #
        # # Predicting with the regressed weights
        # for data_type in ["validation", "test"]:
        #     correct_count = 0
        #     real_least_squares_correct_count = 0
        #     posterior_features = features_dict[data_type][0]
        #     activation_features = features_dict[data_type][1]
        #     lst_weights = targets_dict[data_type][0]
        #     feature_vectors = get_feature_vectors(data_type)
        #     weights_predicted = best_model.predict(X=feature_vectors)
        #     routing_matrix = routing_matrices_dict[data_type]
        #     posteriors_tensor = posterior_tensors_dict[data_type]
        #     multi_path_indices_ = multi_path_indices_dict[data_type]
        #     data_obj = data_objects_dict[data_type]
        #     for idx, multi_path_idx in enumerate(multi_path_indices_):
        #         true_label = data_obj.labelList[multi_path_idx]
        #         routing_vector = routing_matrix[multi_path_idx]
        #         posterior_matrix = posteriors_tensor[multi_path_idx, :, :]
        #         weights = weights_predicted[idx, :]
        #         lst_weights_ = lst_weights[idx, :]
        #         sparse_posteriors_matrix = posterior_matrix * np.expand_dims(routing_vector, axis=0)
        #         lst_squares_posterior_regressed = sparse_posteriors_matrix @ weights
        #         lst_squares_regressor_predicted_label = np.argmax(lst_squares_posterior_regressed)
        #         correct_count += int(lst_squares_regressor_predicted_label == true_label)
        #         lst_squares_posterior = sparse_posteriors_matrix @ lst_weights_
        #         lst_squares_predicted_label = np.argmax(lst_squares_posterior)
        #         real_least_squares_correct_count += int(lst_squares_predicted_label == true_label)
        #     multi_path_accuracy_lst_squares_regression = (single_path_correct_counts_dict[data_type] + correct_count) \
        #                                                  / routing_matrix.shape[0]
        #     multi_path_accuracy_lst_squares = (single_path_correct_counts_dict[
        #                                            data_type] + real_least_squares_correct_count) \
        #                                       / routing_matrix.shape[0]
        #     print("Regression Result DataType:{0} Accuracy:{1}".format(data_type,
        #                                                                multi_path_accuracy_lst_squares_regression))
        #     print("Actual Least Squares Result DataType:{0} Accuracy:{1}".format(data_type,
        #                                                                          multi_path_accuracy_lst_squares))
        # print("X")

        # Model each weight with different regressors
        models = []
        for leaf_idx, leaf_node in enumerate(self.leafNodes):
            print("TRAINING FOR LEAF:{0}".format(leaf_idx))
            routed_to_leaf_indicator_vector = validation_routing_matrix[:, leaf_idx]
            positive_indices = np.nonzero(routed_to_leaf_indicator_vector == 1)[0]
            # Get feature vectors
            x_leaf_train = x_train[positive_indices, :]
            # Get target_variable
            y_leaf_train = y_train[positive_indices, leaf_idx]
            # Fit the regressor now
            feature_dim = x_leaf_train.shape[1]
            pca = PCA()
            regressor_tpl, param_grid = get_mlp_regressor()
            pipe = Pipeline(steps=[("pca", pca), regressor_tpl])
            step = max(1, int((feature_dim - 5) / 50))
            # Hyperparameter grid
            # pca__n_components = [d for d in range(5, feature_dim, step)]
            # pca__n_components.append(feature_dim)
            pca__n_components = [feature_dim]
            param_grid["pca__n_components"] = pca__n_components
            grid_search = GridSearchCV(pipe, param_grid, iid=False, cv=5, n_jobs=8, refit=True, verbose=10)
            grid_search.fit(X=x_leaf_train, y=y_leaf_train)
            best_model = grid_search.best_estimator_
            models.append(best_model)
            print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
            print(grid_search.best_params_)

        # Evaluate the models
        # Predicting with the regressed weights
        for data_type in ["validation", "test"]:
            correct_count = 0
            real_least_squares_correct_count = 0
            posterior_features = features_dict[data_type][0]
            activation_features = features_dict[data_type][1]
            lst_weights = targets_dict[data_type][0]
            feature_vectors = get_feature_vectors(data_type)
            routing_matrix = routing_matrices_dict[data_type]
            posteriors_tensor = posterior_tensors_dict[data_type]
            multi_path_indices_ = multi_path_indices_dict[data_type]
            data_obj = data_objects_dict[data_type]
            for idx, multi_path_idx in enumerate(multi_path_indices_):
                true_label = data_obj.labelList[multi_path_idx]
                routing_vector = routing_matrix[multi_path_idx]
                posterior_matrix = posteriors_tensor[multi_path_idx, :, :]
                feature_vector = feature_vectors[idx]
                lst_weights_ = lst_weights[idx, :]
                weights_predicted = []
                for leaf_id in range(len(self.leafNodes)):
                    if routing_vector[leaf_id] == 0:
                        weights_predicted.append(0.0)
                        continue
                    weight = models[leaf_id].predict(X=np.expand_dims(feature_vector, axis=0))
                    weights_predicted.append(weight[0])
                weights_predicted = np.array(weights_predicted)
                sparse_posteriors_matrix = posterior_matrix * np.expand_dims(routing_vector, axis=0)
                lst_squares_posterior_regressed = sparse_posteriors_matrix @ weights_predicted
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
