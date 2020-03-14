from simple_tf.cign.fast_tree import FastTreeNetwork

network_id = 452
network_name = "FashionNet_Lite"

output_names = ["activations", "branch_probs", "label_tensor", "posterior_probs", "branching_feature",
                "pre_branch_feature"]
used_output_names = ["activations", "label_tensor", "pre_branch_feature", "branching_feature"]
iterations = sorted([43680, 44160, 44640, 45120, 45600, 46080, 46560, 47040, 47520, 48000])


class DatasetLinkingAlgorithm:
    def __init__(self):
        pass

    @staticmethod
    def run():
        data_dict = {}
        for iteration in iterations:
            if iteration == iterations[-1]:
                break
            network = FastTreeNetwork.get_mock_tree(degree_list=[2, 2], network_name=network_name)
            routing_data = network.load_routing_info(run_id=network_id, iteration=iteration, data_type="test",
                                                     output_names=output_names)
            data_dict[iteration] = routing_data




