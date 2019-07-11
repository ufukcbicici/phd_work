import numpy as np
import scipy

from algorithms.softmax_compresser import SoftmaxCompresser
from auxillary.general_utility_funcs import UtilityFuncs
from scipy.stats import expon
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

for leaf_index in {3, 4, 5, 6}:
    file_name = "npz_node_{0}_all_data".format(leaf_index)
    data = UtilityFuncs.load_npz(file_name=file_name)
    training_features = data["training_features"]
    training_labels = data["training_labels"]
    test_features = data["test_features"]
    test_labels = data["test_labels"]
    training_posteriors_compressed = data["training_posteriors_compressed"]
    training_one_hot_labels_compressed = data["training_one_hot_labels_compressed"]
    test_posteriors_compressed = data["test_posteriors_compressed"]
    test_one_hot_labels_compressed = data["test_one_hot_labels_compressed"]

    test_accuracy_full = \
        SoftmaxCompresser.calculate_compressed_accuracy(posteriors=test_posteriors_compressed,
                                                        one_hot_labels=test_one_hot_labels_compressed)

    exponential_distribution = scipy.stats.expon(scale=100)
    all_regularizer_values = exponential_distribution.rvs(100).tolist()
    lesser_than_one = np.linspace(0.00001, 1.0, 11)
    all_regularizer_values.extend(lesser_than_one)
    all_regularizer_values.extend([10, 100, 1000, 10000])
    regularizer_dict = {"C": all_regularizer_values}
    svm = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                    multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                    verbose=0)
    grid_search = GridSearchCV(estimator=svm, param_grid=regularizer_dict, cv=5, n_jobs=1, scoring=None, refit=True)
    grid_search.fit(X=training_features, y=training_labels)
    best_svm = grid_search.best_estimator_
    hyperplanes = np.transpose(best_svm.coef_)
    normalized_hyperplanes = np.copy(hyperplanes)
    biases = best_svm.intercept_
    for col in range(hyperplanes.shape[1]):
        magnitude = np.asscalar(np.linalg.norm(hyperplanes[:, col]))
        normalized_hyperplanes[:, col] = hyperplanes[:, col] / magnitude
    confidences = best_svm.decision_function(X=test_features)
    confidences_manual = np.dot(test_features, hyperplanes) + biases
    confidences_manual_normalized = np.dot(test_features, normalized_hyperplanes) + biases
    assert np.allclose(confidences, confidences_manual)
    print("Baseline Accuracy:{0}".format(test_accuracy_full))
    print("Svm Accuracy:{0}".format(best_svm.score(X=test_features, y=test_labels)))
    print("X")
