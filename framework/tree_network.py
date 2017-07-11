from framework.network import Network
from framework.network_node import NetworkNode


class TreeNetwork(Network):
    def __init__(self, run_id, dataset, parameter_file, tree_degree, list_of_node_builder_functions):
        super().__init__(run_id, dataset, parameter_file)
        self.treeDegree = tree_degree
        self.treeDepth = len(list_of_node_builder_functions)
        self.depthsToNodesDict = {}
        self.nodesToDepthsDict = {}
        self.nodeBuilderFunctions = list_of_node_builder_functions

    def build_network(self):
        curr_index = 0
        # Topologically build the node ordering
        for depth in range(0, self.treeDepth):
            node_count_in_depth = pow(self.treeDegree, depth)
            for i in range(0, node_count_in_depth):
                is_root = depth == 0
                is_leaf = depth == (self.treeDepth - 1)
                node = NetworkNode(index=curr_index, containing_network=self, is_root=is_root,
                                   is_leaf=is_leaf)
                self.nodes[node.index] = node
                if is_leaf:
                    self.leafNodes.append(node)
                self.add_node_to_depth(depth=depth, node=node)
                if not is_root:
                    parent_index = self.get_parent_index(node_index=curr_index)
                    self.dag.add_edge(parent=self.nodes[parent_index], child=node)
                else:
                    self.dag.add_node(node=node)
                curr_index += 1
        self.topologicalSortedNodes = self.dag.get_topological_sort()
        # Build the symbolic graphs of the nodes
        for node in self.topologicalSortedNodes:
            node_depth = self.nodesToDepthsDict[node]
            self.nodeBuilderFunctions[node_depth](node=node)

    # Private methods
    def get_parent_index(self, node_index):
        parent_index = int((node_index - 1) / self.treeDegree)
        return parent_index

    def add_node_to_depth(self, depth, node):
        if not (depth in self.depthsToNodesDict):
            self.depthsToNodesDict[depth] = []
        self.depthsToNodesDict[depth].append(node)
        self.nodesToDepthsDict[node] = depth
