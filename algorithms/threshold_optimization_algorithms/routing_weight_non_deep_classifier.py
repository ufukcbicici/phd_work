from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from algorithms.threshold_optimization_algorithms.routing_weights_deep_softmax_regressor import \
    RoutingWeightDeepSoftmaxRegressor
import tensorflow as tf
import numpy as np


class RoutingWeightNonDeepClassifier(RoutingWeightDeepSoftmaxRegressor):
    def __init__(self, network, validation_routing_matrix, test_routing_matrix, validation_data, test_data, layers,
                 l2_lambda, batch_size, max_iteration, use_multi_path_only=False):
        super().__init__(network, validation_routing_matrix, test_routing_matrix, validation_data, test_data, layers,
                         l2_lambda, batch_size, max_iteration, use_multi_path_only)

    def get_rdf_training_grid(self):
        n_estimators = [100, 250, 500, 1000]
        bootstrap = [False, True]
        max_depth = [5, 10, 15, 20, 25, 30, 40]
        # verbose = [100]
        class_weight = [None, "balanced"]
        tuned_parameters = [{"n_estimators": n_estimators, "bootstrap": bootstrap,
                             "max_depth": max_depth, "class_weight": class_weight}]
        return tuned_parameters

    def run(self):
        # RDF
        # Prepare training data
        hyperparameters = self.get_rdf_training_grid()
        rdf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        grid_search = GridSearchCV(estimator=rdf, param_grid=hyperparameters, cv=6,
                                   n_jobs=8, scoring=None, refit=True, verbose=5)
        grid_search.fit(X=self.multiPathDataDict["validation"].X, y=self.multiPathDataDict["validation"].y)
        best_model = grid_search.best_estimator_
        val_accuracy = best_model.score(X=self.multiPathDataDict["validation"].X,
                                        y=self.multiPathDataDict["validation"].y)
        test_accuracy = best_model.score(X=self.multiPathDataDict["test"].X,
                                        y=self.multiPathDataDict["test"].y)
        print("val_accuracy={0}".format(val_accuracy))
        print("test_accuracy={0}".format(test_accuracy))
