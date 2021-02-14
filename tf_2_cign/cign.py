from collections import deque
from collections import Counter
import numpy as np
# import tensorflow as tf

from auxillary.dag_utilities import Dag
from simple_tf.uncategorized.node import Node


class Cign:
    def __init__(self, input_dims, node_degrees):
        self.dagObject = Dag()
        self.nodes = {}
        self.degreeList = node_degrees
        self.networkDepth = len(self.degreeList)
        # self.inputs = tf.keras.Input(shape=input_dims, name="input")
        # self.labels = tf.keras.Input(shape=None, name="labels", dtype=tf.int32)
        # self.batchSize = tf.keras.Input(shape=None, name="batchSize", dtype=tf.int32)
        # Hyper-parameters
        self.routingSoftmaxDecays = {}

    def build_tree(self):
        # Create itself
        curr_index = 0
        is_leaf = 0 == self.networkDepth
        root_node = Node(index=curr_index, depth=0, is_root=True, is_leaf=is_leaf)
        self.dagObject.add_node(node=root_node)
        self.nodes[curr_index] = root_node
        d = deque()
        d.append(root_node)
        # Create children if not leaf
        while len(d) > 0:
            # Dequeue
            curr_node = d.popleft()
            if not curr_node.isLeaf:
                for i in range(self.degreeList[curr_node.depth]):
                    new_depth = curr_node.depth + 1
                    is_leaf = new_depth == self.networkDepth
                    curr_index += 1
                    child_node = Node(index=curr_index, depth=new_depth, is_root=False, is_leaf=is_leaf)
                    self.nodes[curr_index] = child_node
                    self.dagObject.add_edge(parent=curr_node, child=child_node)
                    d.append(child_node)

    def build_network(self):
        self.build_tree()










        # threshold_name = self.get_variable_name(name="threshold", node=root_node)
        # root_node.probabilityThreshold = tf.placeholder(name=threshold_name, dtype=tf.float32)
        # softmax_decay_name = self.get_variable_name(name="softmax_decay", node=root_node)
        # root_node.softmaxDecay = tf.placeholder(name=softmax_decay_name, dtype=tf.float32)
        # self.dagObject.add_node(node=root_node)
        # self.nodes[curr_index] = root_node
        # d = deque()
        # d.append(root_node)
        # # Create children if not leaf
        # while len(d) > 0:
        #     # Dequeue
        #     curr_node = d.popleft()
        #     if not curr_node.isLeaf:
        #         for i in range(self.degreeList[curr_node.depth]):
        #             new_depth = curr_node.depth + 1
        #             is_leaf = new_depth == (self.depth - 1)
        #             curr_index += 1
        #             child_node = Node(index=curr_index, depth=new_depth, is_root=False, is_leaf=is_leaf)
        #             if not child_node.isLeaf:
        #                 threshold_name = self.get_variable_name(name="threshold", node=child_node)
        #                 child_node.probabilityThreshold = tf.placeholder(name=threshold_name, dtype=tf.float32)
        #                 softmax_decay_name = self.get_variable_name(name="softmax_decay", node=child_node)
        #                 child_node.softmaxDecay = tf.placeholder(name=softmax_decay_name, dtype=tf.float32)
        #             self.nodes[curr_index] = child_node
        #             self.dagObject.add_edge(parent=curr_node, child=child_node)
        #             d.append(child_node)
        # nodes_per_level_dict = {}
        # for node in self.nodes.values():
        #     if node.depth not in nodes_per_level_dict:
        #         nodes_per_level_dict[node.depth] = []
        #     nodes_per_level_dict[node.depth].append(node)
        # for level in nodes_per_level_dict.keys():
        #     self.orderedNodesPerLevel[level] = sorted(nodes_per_level_dict[level], key=lambda n: n.index)
        # self.topologicalSortedNodes = self.dagObject.get_topological_sort()
        # self.innerNodes = [node for node in self.topologicalSortedNodes if not node.isLeaf]
        # self.leafNodes = [node for node in self.topologicalSortedNodes if node.isLeaf]
        # self.innerNodes = sorted(self.innerNodes, key=lambda nd: nd.index)
        # self.leafNodes = sorted(self.leafNodes, key=lambda nd: nd.index)


        # # Build the tree topologically and create the Tensorflow placeholders
        # self.build_tree()
        # # Build symbolic networks
        # self.isBaseline = len(self.topologicalSortedNodes) == 1
        # # Disable some properties if we are using a baseline
        # if self.isBaseline:
        #     GlobalConstants.USE_INFO_GAIN_DECISION = False
        #     GlobalConstants.USE_CONCAT_TRICK = False
        #     GlobalConstants.USE_PROBABILITY_THRESHOLD = False
        # # Build all symbolic networks in each node
        # for node in self.topologicalSortedNodes:
        #     print("Building Node {0}".format(node.index))
        #     self.nodeBuildFuncs[node.depth](network=self, node=node)
        # # Build the residue loss
        # # self.build_residue_loss()
        # # Record all variables into the variable manager (For backwards compatibility)
        # # self.variableManager.get_all_node_variables()
        # self.dbName = DbLogger.log_db_path[DbLogger.log_db_path.rindex("/") + 1:]
        # print(self.dbName)
        # self.nodeCosts = {node.index: node.macCost for node in self.topologicalSortedNodes}
        # # Build main classification loss
        # self.build_main_loss()
        # # Build information gain loss
        # self.build_decision_loss()
        # # Build regularization loss
        # self.build_regularization_loss()
        # # Final Loss
        # self.finalLoss = self.mainLoss + self.regularizationLoss + self.decisionLoss
        # if not GlobalConstants.USE_MULTI_GPU:
        #     self.build_optimizer()
        # self.prepare_evaluation_dictionary()