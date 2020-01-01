from simple_tf.cign.fast_tree import FastTreeNetwork


class CombinatorialRoutingOptimizer:
    def __init__(self, network_name, run_id, iteration, degree_list, original_images, augmented_images,
                 output_names):
        self.network = FastTreeNetwork.get_mock_tree(degree_list=degree_list, network_name=network_name)
        self.innerNodes = [node for node in self.network.topologicalSortedNodes if not node.isLeaf]
        self.leafNodes = [node for node in self.network.topologicalSortedNodes if node.isLeaf]
        self.innerNodes = sorted(self.innerNodes, key=lambda node: node.index)
        self.leafNodes = sorted(self.leafNodes, key=lambda node: node.index)
        self.routingData = self.network.load_routing_info(run_id=run_id, iteration=iteration, data_type="test",
                                                          output_names=output_names)
        self.branchProbs = self.routingData.get_dict("branch_probs")
        self.posteriors = self.routingData.get_dict("posterior_probs")
        self.labelsVector = self.routingData.labelList

    def calculate_best_routes(self):
        # for idx in range(self.posteriors.shape[0]):
        #     curr_node = self.network.topologicalSortedNodes[0]
        #     total_path_entropy = 0
        #     route = []
        #     while True:
        #         route.append(curr_node.index)
        #         if curr_node.isLeaf:
        #             self.totalPathEntropies.append(total_path_entropy)
        #             self.routesPerSample.append(route)
        #             break
        #         routing_distribution = self.branchProbs[curr_node.index][idx]
        #         entropy = UtilityFuncs.calculate_distribution_entropy(distribution=routing_distribution)
        #         total_path_entropy += entropy
        #         arg_max_child_index = np.argmax(routing_distribution)
        #         child_nodes = self.network.dagObject.children(node=curr_node)
        #         child_nodes = sorted(child_nodes, key=lambda c_node: c_node.index)
        #         curr_node = child_nodes[arg_max_child_index]