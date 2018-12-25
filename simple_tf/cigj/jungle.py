import matplotlib.pyplot as plt
import tensorflow as tf

from auxillary.dag_utilities import Dag
from auxillary.general_utility_funcs import UtilityFuncs
from simple_tf.cigj.jungle_node import JungleNode
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.global_params import GlobalConstants
from simple_tf.info_gain import InfoGainLoss


class Jungle(FastTreeNetwork):
    def __init__(self, node_build_funcs, h_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list,
                 dataset):
        assert len(node_build_funcs) == len(h_funcs) + 1
        super().__init__(node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset)
        curr_index = 0
        self.depthToNodesDict = {}
        self.hFuncs = h_funcs
        self.currentGraph = tf.get_default_graph()
        self.batchIndices = tf.range(self.batchSize)
        # self.decisionNoiseFactor = tf.placeholder(name="decision_noise_factor", dtype=tf.float32)
        # Create Trellis structure. Add a h node to every non-root and non-leaf layer.
        degree_list = [degree if depth == len(degree_list) - 1 else degree + 1 for depth, degree in
                       enumerate(degree_list)]
        assert degree_list[0] == 2
        assert degree_list[-1] == 1
        for depth, num_of_nodes in enumerate(degree_list):
            # root node, F_nodes, leaf nodes and H_node
            for index_in_depth in range(num_of_nodes):
                if depth < len(degree_list) - 1 and index_in_depth == num_of_nodes - 1:
                    node_type = NodeType.h_node
                elif depth == 0 and index_in_depth == 0:
                    assert num_of_nodes == 2
                    node_type = NodeType.root_node
                elif depth == len(degree_list) - 1:
                    node_type = NodeType.leaf_node
                else:
                    node_type = NodeType.f_node
                curr_node = JungleNode(index=curr_index, depth=depth, node_type=node_type)
                self.nodes[curr_index] = curr_node
                curr_index += 1
                if depth not in self.depthToNodesDict:
                    self.depthToNodesDict[depth] = []
                self.depthToNodesDict[depth].append(curr_node)
        # Build network as a DAG
        self.build_network()
        self.print_trellis_structure()

    def get_session(self):
        sess = tf.Session(graph=self.currentGraph)
        return sess

    def build_network(self):
        # Each H node will have the F nodes and the root node in the same layer and the H node in the previous layer
        # as the parents.
        # Each F node and leaf node have the H node in the previous layer as the parent.
        self.dagObject = Dag()
        for node in self.nodes.values():
            print(node.nodeType)
            if node.nodeType == NodeType.root_node:
                continue
            elif node.nodeType == NodeType.f_node or node.nodeType == NodeType.leaf_node:
                parent_h_nodes = [candidate_node for candidate_node in self.nodes.values()
                                  if candidate_node.depth == node.depth - 1
                                  and candidate_node.nodeType == NodeType.h_node]
                assert len(parent_h_nodes) == 1
                parent_h_node = parent_h_nodes[0]
                self.dagObject.add_edge(parent=parent_h_node, child=node)
            else:
                assert node.nodeType == NodeType.h_node
                parent_nodes = [candidate_node for candidate_node in self.nodes.values()
                                if (candidate_node.depth == node.depth
                                and (candidate_node.nodeType == NodeType.f_node or
                                     candidate_node.nodeType == NodeType.root_node)) or
                                (candidate_node.depth == node.depth - 1 and candidate_node.nodeType == NodeType.h_node)]
                for parent_node in parent_nodes:
                    self.dagObject.add_edge(parent=parent_node, child=node)
        self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        # Build auxillary variables
        self.thresholdFunc(network=self)
        # Build node computational graphs
        for node in self.topologicalSortedNodes:
            if node.depth > 3:
                continue
            if node.nodeType == NodeType.root_node or node.nodeType == NodeType.f_node or \
                    node.nodeType == NodeType.leaf_node:
                self.nodeBuildFuncs[node.depth](node=node, network=self)
                assert node.F_output is not None and node.H_output is None
                node.evalDict[UtilityFuncs.get_variable_name(name="F_output", node=node)] = node.F_output
            elif node.nodeType == NodeType.h_node:
                self.hFuncs[node.depth](node=node, network=self)
                assert node.F_output is not None and node.H_output is not None
                node.evalDict[UtilityFuncs.get_variable_name(name="F_output", node=node)] = node.F_output
                node.evalDict[UtilityFuncs.get_variable_name(name="H_output", node=node)] = node.H_output
        # Build the network eval dict
        self.evalDict = {}
        for node in self.topologicalSortedNodes:
            for k, v in node.evalDict.items():
                assert k not in self.evalDict
                self.evalDict[k] = v

    def stitch_samples(self, node):
        assert node.nodeType == NodeType.h_node
        parents = self.dagObject.parents(node=node)
        # Layer 0 h_node. This receives non-partitioned, complete minibatch from the root node. No stitching needed.
        if len(parents) == 1:
            assert parents[0].nodeType == NodeType.root_node and node.depth == 0
            node.F_input = parents[0].F_output
            node.H_input = None
        # Need stitching
        else:
            # Get all F nodes in the same layer
            parent_f_nodes = [f_node for f_node in self.dagObject.parents(node=node)
                              if f_node.nodeType == NodeType.f_node]
            parent_h_nodes = [h_node for h_node in self.dagObject.parents(node=node)
                              if h_node.nodeType == NodeType.h_node]
            assert len(parent_h_nodes) == 1
            parent_h_node = parent_h_nodes[0]
            parent_f_nodes = sorted(parent_f_nodes, key=lambda f_node: f_node.index)
            assert all([f_node.H_output is None for f_node in parent_f_nodes])
            f_inputs = [node.F_output for node in parent_f_nodes]
            node.F_input = tf.dynamic_stitch(indices=parent_h_node.conditionIndices, data=f_inputs)
            node.H_input = tf.dynamic_stitch(indices=parent_h_node.conditionIndices, data=parent_h_node.H_output)

    def apply_decision(self, node, branching_feature):
        assert node.nodeType == NodeType.h_node
        node.H_output = branching_feature
        node_degree = self.degreeList[node.depth + 1]
        if node_degree > 1:
            # Step 1: Create Hyperplanes
            ig_feature_size = node.H_output.get_shape().as_list()[-1]
            hyperplane_weights = tf.Variable(
                tf.truncated_normal([ig_feature_size, node_degree], stddev=0.1, seed=GlobalConstants.SEED,
                                    dtype=GlobalConstants.DATA_TYPE),
                name=UtilityFuncs.get_variable_name(name="hyperplane_weights", node=node))
            hyperplane_biases = tf.Variable(tf.constant(0.0, shape=[node_degree], dtype=GlobalConstants.DATA_TYPE),
                                            name=UtilityFuncs.get_variable_name(name="hyperplane_biases", node=node))
            if GlobalConstants.USE_BATCH_NORM_BEFORE_BRANCHING:
                node.H_output = tf.layers.batch_normalization(inputs=node.H_output,
                                                              momentum=GlobalConstants.BATCH_NORM_DECAY,
                                                              training=tf.cast(self.isTrain, tf.bool))
            # Step 2: Calculate the distribution over the computation units (F nodes in the same layer, p(F|x)
            activations = tf.matmul(node.H_output, hyperplane_weights) + hyperplane_biases
            node.activationsDict[node.index] = activations
            decayed_activation = node.activationsDict[node.index] / tf.reshape(node.softmaxDecay, (1,))
            p_F_given_x = tf.nn.softmax(decayed_activation)
            p_c_given_x = self.oneHotLabelTensor
            node.infoGainLoss = InfoGainLoss.get_loss(p_n_given_x_2d=p_F_given_x, p_c_given_x_2d=p_c_given_x,
                                                      balance_coefficient=self.informationGainBalancingCoefficient)
            # Step 3: Select argmax_F p(F|x)
            indices_tensor = tf.argmax(p_F_given_x, axis=1, output_type=tf.int32)
            # Step 4: Apply partitioning for corresponding F nodes in the same layer.
            node.conditionIndices = tf.dynamic_partition(data=self.batchIndices, partitions=indices_tensor,
                                                         num_partitions=node_degree)
            node.F_output = tf.dynamic_partition(data=node.F_input, partitions=indices_tensor,
                                                 num_partitions=node_degree)
            node.H_output = tf.dynamic_partition(data=node.H_output, partitions=indices_tensor,
                                                 num_partitions=node_degree)
            node.labelTensor = tf.dynamic_partition(data=self.labelTensor, partitions=indices_tensor,
                                                    num_partitions=node_degree)
            # Reporting
            node.evalDict[UtilityFuncs.get_variable_name(name="branching_feature", node=node)] = branching_feature
            node.evalDict[UtilityFuncs.get_variable_name(name="activations", node=node)] = activations
            node.evalDict[UtilityFuncs.get_variable_name(name="decayed_activation", node=node)] = decayed_activation
            node.evalDict[UtilityFuncs.get_variable_name(name="softmax_decay", node=node)] = node.softmaxDecay
            node.evalDict[UtilityFuncs.get_variable_name(name="info_gain", node=node)] = node.infoGainLoss
            node.evalDict[UtilityFuncs.get_variable_name(name="p(n|x)", node=node)] = p_F_given_x
            node.evalDict[UtilityFuncs.get_variable_name(name="condition_indices", node=node)] = node.conditionIndices
            node.evalDict[UtilityFuncs.get_variable_name(name="labelTensor", node=node)] = node.labelTensor
            node.evalDict[UtilityFuncs.get_variable_name(name="indices_tensor", node=node)] = indices_tensor
        else:
            node.conditionIndices = [self.batchIndices]
            node.F_output = [node.F_input]
            node.H_output = [node.H_output]
            node.labelTensor = [self.labelTensor]

    def apply_loss_jungle(self, node, final_feature):
        assert len(final_feature.get_shape().as_list()) == 2
        final_feature_dim = final_feature.get_shape().as_list()[-1]
        fc_softmax_weights = tf.Variable(
            tf.truncated_normal([final_feature_dim, self.labelCount], stddev=0.1, seed=GlobalConstants.SEED,
                                dtype=GlobalConstants.DATA_TYPE),
            name=UtilityFuncs.get_variable_name(name="fc_softmax_weights", node=node))
        fc_softmax_biases = tf.Variable(tf.constant(0.1, shape=[self.labelCount],
                                                    dtype=GlobalConstants.DATA_TYPE),
                                        name=UtilityFuncs.get_variable_name(name="fc_softmax_biases", node=node))
        self.apply_loss(node=node, final_feature=final_feature, softmax_weights=fc_softmax_weights,
                        softmax_biases=fc_softmax_biases)
        assert len(node.lossList) == 1
        node.F_output = node.lossList[0]

    def mask_input_nodes(self, node):
        if node.nodeType == NodeType.root_node:
            node.F_input = self.dataTensor
            node.H_input = None
            node.labelTensor = self.labelTensor
            node.isOpenIndicatorTensor = tf.constant(value=1.0, dtype=tf.float32)
            # For reporting
            node.evalDict[self.get_variable_name(name="sample_count", node=node)] = tf.size(node.labelTensor)
            node.evalDict[self.get_variable_name(name="is_open", node=node)] = node.isOpenIndicatorTensor
        elif node.nodeType == NodeType.f_node or node.nodeType == NodeType.leaf_node:
            # raise NotImplementedError()
            parents = self.dagObject.parents(node=node)
            assert len(parents) == 1 and parents[0].nodeType == NodeType.h_node
            parent_node = parents[0]
            sibling_nodes = [node for node in self.depthToNodesDict[node.depth]
                             if node.nodeType == NodeType.f_node or node.nodeType == NodeType.leaf_node]
            sibling_nodes = {node.index: order_index for order_index, node in
                               enumerate(sorted(sibling_nodes, key=lambda c_node: c_node.index))}
            sibling_order_index = sibling_nodes[node.index]
            node.F_input = parent_node.F_output[sibling_order_index]
            node.H_input = parent_node.H_output[sibling_order_index]
            node.labelTensor = parent_node.labelTensor[sibling_order_index]

    def get_softmax_decays(self, feed_dict, iteration, update):
        for node in self.topologicalSortedNodes:
            if node.nodeType != NodeType.h_node:
                continue
            # Decay for Softmax
            decay = node.softmaxDecayCalculator.value
            if update:
                feed_dict[node.softmaxDecay] = decay
                UtilityFuncs.print("{0} value={1}".format(node.softmaxDecayCalculator.name, decay))
                # Update the Softmax Decay
                node.softmaxDecayCalculator.update(iteration=iteration + 1)
            else:
                feed_dict[node.softmaxDecay] = GlobalConstants.SOFTMAX_TEST_TEMPERATURE

    def prepare_feed_dict(self, minibatch, iteration, use_threshold, is_train, use_masking, batch_size):
        feed_dict = {self.dataTensor: minibatch.samples,
                     self.labelTensor: minibatch.labels,
                     self.indicesTensor: minibatch.indices,
                     self.oneHotLabelTensor: minibatch.one_hot_labels,
                     # self.globalCounter: iteration,
                     self.weightDecayCoeff: GlobalConstants.WEIGHT_DECAY_COEFFICIENT,
                     self.decisionWeightDecayCoeff: GlobalConstants.DECISION_WEIGHT_DECAY_COEFFICIENT,
                     # self.isDecisionPhase: int(is_decision_phase),
                     self.isTrain: int(is_train),
                     self.informationGainBalancingCoefficient: GlobalConstants.INFO_GAIN_BALANCE_COEFFICIENT,
                     self.iterationHolder: iteration,
                     self.batchSize: batch_size}
        if is_train:
            feed_dict[self.classificationDropoutKeepProb] = GlobalConstants.CLASSIFICATION_DROPOUT_PROB
            if not self.isBaseline:
                self.get_softmax_decays(feed_dict=feed_dict, iteration=iteration, update=True)
                # self.get_decision_dropout_prob(feed_dict=feed_dict, iteration=iteration, update=True)
                feed_dict[self.decisionDropoutKeepProb] = GlobalConstants.DECISION_DROPOUT_PROB
                self.get_decision_weight(feed_dict=feed_dict, iteration=iteration, update=True)
        else:
            feed_dict[self.classificationDropoutKeepProb] = 1.0
            if not self.isBaseline:
                self.get_softmax_decays(feed_dict=feed_dict, iteration=1000000, update=False)
                feed_dict[self.decisionDropoutKeepProb] = 1.0
                self.get_decision_weight(feed_dict=feed_dict, iteration=iteration, update=False)
        return feed_dict

    # For debugging
    def print_trellis_structure(self):
        fig, ax = plt.subplots()
        # G = self.dagObject.dagObject
        node_radius = 0.05
        node_circles = []
        node_positions = {}
        # Draw Nodes as Vertices (Circles)
        for curr_depth in range(self.depth):
            nodes_of_curr_depth = self.depthToNodesDict[curr_depth]
            if len(nodes_of_curr_depth) > 1:
                horizontal_step_size = (1.0 - 2 * node_radius) / float(len(nodes_of_curr_depth) - 1.0)
                vertical_step_size = (1.0 - 2 * node_radius) / float(len(self.depthToNodesDict) - 1.0)
            else:
                horizontal_step_size = 0.0
                vertical_step_size = 0.0
            for index_in_depth, node in enumerate(nodes_of_curr_depth):
                if node.nodeType == NodeType.root_node:
                    node_circles.append(plt.Circle((0.5, 1.0 - node_radius), node_radius, color='r'))
                    node_positions[node] = (0.5, 1.0 - node_radius)
                elif node.nodeType == NodeType.leaf_node:
                    node_circles.append(plt.Circle((0.5, node_radius), node_radius, color='y'))
                    node_positions[node] = (0.5, node_radius)
                elif node.nodeType == NodeType.f_node:
                    node_circles.append(plt.Circle((node_radius + index_in_depth * horizontal_step_size,
                                                    1.0 - node_radius - curr_depth * vertical_step_size), node_radius,
                                                   color='b'))
                    node_positions[node] = (node_radius + index_in_depth * horizontal_step_size,
                                            1.0 - node_radius - curr_depth * vertical_step_size)
                elif node.nodeType == NodeType.h_node:
                    node_circles.append(plt.Circle((node_radius + index_in_depth * horizontal_step_size,
                                                    1.0 - node_radius - curr_depth * vertical_step_size), node_radius,
                                                   color='g'))
                    node_positions[node] = (node_radius + index_in_depth * horizontal_step_size,
                                            1.0 - node_radius - curr_depth * vertical_step_size)
                else:
                    raise Exception("Unknown node type.")
        for circle in node_circles:
            ax.add_artist(circle)
        # Draw Edges as Arrows
        for edge in self.dagObject.get_edges():
            source = edge[0]
            destination = edge[1]
            ax.arrow(node_positions[source][0], node_positions[source][1],
                     node_positions[destination][0] - node_positions[source][0],
                     node_positions[destination][1] - node_positions[source][1],
                     head_width=0.01, head_length=0.01, fc='k', ec='k',
                     length_includes_head=True)
        # Draw node texts
        for node in self.topologicalSortedNodes:
            node_pos = node_positions[node]
            ax.text(node_pos[0], node_pos[1], "{0}".format(node.index), fontsize=16, color="c")
        plt.show()
        UtilityFuncs.print("X")

        # def apply_decision(self):
        #     pass
