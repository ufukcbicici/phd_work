import numpy as np
import os
import pickle
from scipy.spatial import distance_matrix

from simple_tf.cign.fast_tree import FastTreeNetwork

network_id = 453
network_name = "FashionNet_Lite"

output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                "pre_branch_feature"]
used_output_names = ["pre_branch_feature", "activations", "branching_feature"]
iterations = sorted([43680, 44160, 44640, 45120, 45600, 46080, 46560, 47040, 47520, 48000])


class DatasetLinkingAlgorithm:
    def __init__(self):
        pass

    @staticmethod
    def calculate_distance_matrices():
        data_dict = {}
        # Load all the data
        for iteration in iterations:
            network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
            routing_data = network.load_routing_info(run_id=network_id, iteration=iteration, data_type="test",
                                                     output_names=output_names)
            data_dict[iteration] = routing_data
        # Link all iterations
        for idx, iteration in enumerate(iterations):
            if iteration == iterations[-1]:
                break
            curr_iteration_data = data_dict[iterations[idx]]
            next_iteration_data = data_dict[iterations[idx + 1]]
            for feature_name in used_output_names:
                curr_iteration_feature_dict = curr_iteration_data.get_dict(feature_name)
                next_iteration_feature_dict = next_iteration_data.get_dict(feature_name)
                assert set(curr_iteration_feature_dict.keys()) == set(next_iteration_feature_dict.keys())
                keys = sorted(list(curr_iteration_feature_dict.keys()))
                for key in keys:
                    distance_matrix_name = "distance_matrix_{0}_node{1}_i{2}_vs_i{3}.sav".format(
                        feature_name, key, iterations[idx], iterations[idx + 1])
                    print("Creating {0}.".format(distance_matrix_name))
                    X_curr = curr_iteration_feature_dict[key]
                    X_next = next_iteration_feature_dict[key]
                    assert X_curr.shape == X_next.shape
                    X_curr = np.reshape(X_curr, newshape=(X_curr.shape[0], np.prod(X_curr.shape[1:])))
                    X_next = np.reshape(X_next, newshape=(X_next.shape[0], np.prod(X_next.shape[1:])))
                    d_M = distance_matrix(X_curr, X_next)
                    directory_path = FastTreeNetwork.get_routing_info_path(network_name=network_name,
                                                                           run_id=network_id, iteration=iteration,
                                                                           data_type="test")
                    pickle.dump(d_M, open(os.path.abspath(os.path.join(directory_path, distance_matrix_name)), "wb"))
                    print("{0} has been created.".format(distance_matrix_name))
            print("X")


def main():
    # compare_gpu_implementation()
    # train_basic_q_learning()
    DatasetLinkingAlgorithm.calculate_distance_matrices()


if __name__ == "__main__":
    main()
