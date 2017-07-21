import tensorflow as tf
from auxillary.constants import ChannelTypes, GlobalInputNames, TreeType
from framework.hard_tree_node import HardTreeNode
from framework.network import Network
from framework.network_channel import NetworkChannel
from framework.network_node import NetworkNode
from framework.node_input_outputs import NetworkInput


class TreeNetwork(Network):
    def __init__(self, run_id, dataset, parameter_file, problem_type,
                 tree_degree, tree_type, list_of_node_builder_functions):
        super().__init__(run_id, dataset, parameter_file, problem_type)
        self.treeDegree = tree_degree
        self.treeDepth = len(list_of_node_builder_functions)
        self.depthsToNodesDict = {}
        self.nodesToDepthsDict = {}
        self.nodeBuilderFunctions = list_of_node_builder_functions
        self.treeType = tree_type
        self.indicatorText = "TreeNetwork"

    # Tensorflow specific code (This should be isolated at some point in future)
    # A node can take from an input from any ancestor node. If the ancestor is its parent, it is directly connected
    # to the parent: The parent has already applied the necessary decision mechanism to the output. If the input is
    # from a non-parent ancestor, than the output is taken from that ancestor, propagated through the ancestor laying
    # in between the path (ancestor, current_node). Each intermediate node, inductively takes the input from its parent,
    # applies decision to it and propagate to the its child. In that case the node producing the output and the interme-
    # diate node are different and this is the only case that happens.
    def add_nodewise_input(self, producer_node, producer_channel, producer_channel_index, dest_node):
        producer_triple = (producer_node, producer_channel, producer_channel_index)
        # Data or label input
        if producer_node is None:
            if producer_channel == ChannelTypes.data_input or producer_channel == ChannelTypes.label_input:
                tensor = None
                with NetworkChannel(node=dest_node, channel=producer_channel) as input_channel:
                    if producer_channel == ChannelTypes.data_input:
                        tensor = self.add_networkwise_input(name=ChannelTypes.data_input.value, channel=input_channel,
                                                            tensor_type=tf.float32)
                    elif producer_channel == ChannelTypes.label_input:
                        tensor = self.add_networkwise_input(name=ChannelTypes.label_input.value, channel=input_channel,
                                                            tensor_type=tf.int64)
                    else:
                        raise NotImplementedError()
                    dest_node.inputs[(producer_node, producer_channel, producer_channel_index)] \
                        = NetworkInput(source_node=None, source_channel=producer_channel,
                                       source_channel_index=producer_channel_index, input_object=tensor,
                                       producer_node=None, producer_channel=producer_channel,
                                       producer_channel_index=producer_channel_index)
                return tensor
            else:
                raise Exception("Only data or label inputs can have no source node.")
        # An input either from a 1)directly from a parent node 2)from an ancestor node.
        # In the second case apply the propagation algorithm.
        else:
            path = self.dag.get_shortest_path(source=producer_node, dest=dest_node)
            last_output = None
            # Assume the path starts from the ancestor to the node
            # Since it is a tree, only a single path exists between the ancestor and the node.
            if len(path) <= 1:
                raise Exception("No path has been found between source node {0} and destination node {1}".
                                format(producer_node.index, dest_node.index))
            for path_node in path:
                # WE ASSUME THAT path_node IS TOPOLOGICALLY ORDERED!!!
                # The source node: The output must already exist before in the ancestor.
                # Output has gone through necessary decision processes.
                if path_node == producer_node:
                    if producer_triple not in path_node.outputs:
                        raise Exception("The output must already exist before.")
                    last_output = path_node.outputs[producer_triple]
                # An ancestor node which on the path [source,dest]. We need to make the desired output of source node
                # pass through the intermediate nodes on the path, such that decisions are applied to them correctly.
                elif path_node != producer_node or path_node != dest_node:
                    # This output is not used by path_node at all.
                    if producer_triple not in path_node.inputs and producer_triple not in path_node.outputs:
                        path_node.inputs[producer_triple] = last_output.produce_input()
                        path_node.create_transfer_channel(producer_node=path_node, producer_channel=producer_channel,
                                                          producer_channel_index=producer_channel_index)
                        last_output = path_node.outputs[producer_triple]
                    # This output is used by the path but for its internal calculations.
                    elif producer_triple in path_node.inputs and producer_triple not in path_node.outputs:
                        path_node.create_transfer_channel(producer_node=path_node,
                                                          producer_channel=producer_channel,
                                                          producer_channel_index=producer_channel_index)
                        last_output = path_node.outputs[producer_triple]
                    # There cannot be an operation which is not input but in output. This is invalid.
                    elif producer_triple not in path_node.inputs and producer_triple in path_node.outputs:
                        raise Exception("Operation {0} is not in the input of the node {1} but in its outputs!".format(
                            producer_triple, path_node.index))
                    # Both in input and output.
                    # Then this output from the ancestor is already being broadcast from this node.
                    else:
                        last_output = path_node.outputs[producer_triple]
                # The node which will finally use the output.
                else:
                    # It shouldn't be in the input dict.
                    if producer_triple in path_node.inputs:
                        raise Exception("The triple {0} must not be in the inputs.".format(producer_triple))
                    else:
                        path_node.inputs[producer_triple] = last_output.produce_input()
                        return path_node.inputs[producer_triple].inputObject

    def apply_decision(self, node, channel, channel_index):
        raise NotImplementedError()

    def build_network(self):
        curr_index = 0
        # Step 1:
        # Topologically build the node ordering
        for depth in range(0, self.treeDepth):
            node_count_in_depth = pow(self.treeDegree, depth)
            for i in range(0, node_count_in_depth):
                is_root = depth == 0
                is_leaf = depth == (self.treeDepth - 1)
                if self.treeType == TreeType.hard:
                    node = HardTreeNode(index=curr_index, containing_network=self, is_root=is_root,
                                        is_leaf=is_leaf)
                else:
                    raise NotImplementedError()
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
        # Add a final, accumulation node, which combines all the losses from all parent nodes.
        accumulation_node = HardTreeNode(index=curr_index, containing_network=self, is_root=False, is_leaf=False)
        self.add_node_to_depth(depth=self.treeDepth, node=accumulation_node)
        for node in self.nodes.values():
            if node.isLeaf:
                self.dag.add_edge(parent=node, child=accumulation_node)
        self.topologicalSortedNodes = self.dag.get_topological_sort()
        # Step 2:
        # Build the complete symbolic graph by building and connectiong the symbolic graphs of the nodes
        for node in self.topologicalSortedNodes:
            node_depth = self.nodesToDepthsDict[node]
            # Add customized operations on the top.
            with tf.variable_scope(node.indicatorText):
                # Build the node-wise ops
                if not node.isAccumulation:
                    self.nodeBuilderFunctions[node_depth](network=self, node=node)
                    # Build weight shrinkage losses.
                    node.attach_shrinkage_losses()
                # If node is leaf, apply the actual loss function. If accumulation, fetch all losses from all nodes
                # in the graph, apply gradient calculations, etc.
                if node.isLeaf or node.isAccumulation:
                    node.attach_loss_eval_channels()

    # Private methods
    def get_parent_index(self, node_index):
        parent_index = int((node_index - 1) / self.treeDegree)
        return parent_index

    def add_node_to_depth(self, depth, node):
        if not (depth in self.depthsToNodesDict):
            self.depthsToNodesDict[depth] = []
        self.depthsToNodesDict[depth].append(node)
        self.nodesToDepthsDict[node] = depth
