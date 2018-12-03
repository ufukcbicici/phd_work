from collections import deque

import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from simple_tf.cign.fast_tree import FastTreeNetwork
from simple_tf.node import Node
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.cigj.jungle_node import JungleNode


class Jungle(FastTreeNetwork):
    def __init__(self, node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset):
        super().__init__(node_build_funcs, grad_func, threshold_func, residue_func, summary_func, degree_list, dataset)
        curr_index = 0
        is_leaf = 0 == (self.depth - 1)
        root_node = Node(index=curr_index, depth=0, is_root=True, is_leaf=is_leaf)
        threshold_name = self.get_variable_name(name="threshold", node=root_node)
        root_node.probabilityThreshold = tf.placeholder(name=threshold_name, dtype=tf.float32)
        softmax_decay_name = self.get_variable_name(name="softmax_decay", node=root_node)
        root_node.softmaxDecay = tf.placeholder(name=softmax_decay_name, dtype=tf.float32)
        self.dagObject.add_node(node=root_node)
        self.nodes[curr_index] = root_node
        self.depthToNodesDict = {}
        d = deque()
        d.append(root_node)
        # Create Trellis structure. Add a h node to every non-root and non-leaf layer.
        degree_list = [degree if depth == 0 or depth == len(degree_list) - 1 else degree + 1 for depth, degree in
                       enumerate(degree_list)]
        assert degree_list[0] == 1
        assert degree_list[-1] == 1
        for depth, num_of_nodes in enumerate(degree_list):
            for i in range(num_of_nodes):
                if depth == 0:
                    node_type = NodeType.root_node
                elif depth == len(degree_list) - 1:
                    node_type = NodeType.leaf_node
                elif i < num_of_nodes - 1:
                    node_type = NodeType.f_node
                elif i == num_of_nodes - 1:
                    node_type = NodeType.h_node
                else:
                    raise Exception("Unknown node type.")
                curr_node = JungleNode(index=curr_index, depth=depth, node_type=node_type)
                self.nodes[curr_index] = curr_node
                curr_index += 1
                if depth not in self.depthToNodesDict:
                    self.depthToNodesDict[depth] = []
                self.depthToNodesDict[depth].append(curr_node)
                # Make this node a child to every node in the previous layer
                if depth - 1 in self.depthToNodesDict:
                    for parent_node in self.depthToNodesDict[depth - 1]:
                        self.dagObject.add_edge(parent=parent_node, child=curr_node)
                # Decorate node accordingly with its type
                self.decorate_node(node=curr_node)

    def decorate_node(self, node):
        if node.nodeType == NodeType.h_node:
            threshold_name = self.get_variable_name(name="threshold", node=node)
            softmax_decay_name = self.get_variable_name(name="softmax_decay", node=node)
            node.probabilityThreshold = tf.placeholder(name=threshold_name, dtype=tf.float32)
            node.softmaxDecay = tf.placeholder(name=softmax_decay_name, dtype=tf.float32)

    # For debugging
    def print_trellis_structure(self):
        fig, ax = plt.subplots()
        # G = self.dagObject.dagObject
        max_layer_width = max([len(l) for l in self.depthToNodesDict.values()])
        total_depth = len(self.depthToNodesDict)
        vertical_spacing = 50
        horizontal_spacing = 50
        node_radius = 0.05
        total_width = vertical_spacing * (max_layer_width - 1) + 2 * node_radius
        total_height = horizontal_spacing * (total_depth - 1) + 2 * node_radius
        node_circles = []
        node_positions = {}
        edge_arrows = []
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
        plt.show()
        print("X")
        # pos_dict = {}
        # for node in index_sorted_nodes:
        #     if node.index == 0:
        #         pos_dict[node] = (total_width/2, node_dim)
        #     elif node.index == len(index_sorted_nodes) - 1:
        #         pos_dict[node] = (total_width/2, total_height-node_dim)
        #     else:
        #
