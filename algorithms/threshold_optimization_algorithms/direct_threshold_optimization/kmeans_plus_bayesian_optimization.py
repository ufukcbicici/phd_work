import numpy
from sklearn.cluster import KMeans
from auxillary.general_utility_funcs import UtilityFuncs


class KmeansPlusBayesianOptimization:
    def __init__(self):
        pass

    @staticmethod
    def optimize(cluster_count, fc_layers, run_id, network, routing_data, seed):
        train_indices = routing_data.trainingIndices
        test_indices = routing_data.testIndices
        X_vectorized = UtilityFuncs.vectorize_with_gap(routing_data.get_dict("pre_branch_feature")[0])
        X_train = X_vectorized[train_indices]
        X_test = X_vectorized[test_indices]
        kmeans = KMeans(n_clusters=cluster_count).fit(X_train)
        print("X")
