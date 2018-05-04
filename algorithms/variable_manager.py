import tensorflow as tf
import numpy as np

from simple_tf.global_params import GlobalConstants


class VariableManager:
    def __init__(self, network):
        self.network = network
        self.trainableVariables = []
        self.varToNodesDict = {}
        self.nodesToVarsDict = {}

    def trainable_variables(self):
        return self.trainableVariables

    def create_and_add_variable_to_node(self, node, name, initial_value, is_trainable=True):
        tf_variable = tf.Variable(initial_value=initial_value, name=name, trainable=is_trainable)
        self.add_variable_to_node(node=node, tf_variable=tf_variable)
        return tf_variable

    def add_variables_to_node(self, node, tf_variables):
        for variable in tf_variables:
            self.add_variable_to_node(node=node, tf_variable=variable)

    def add_variable_to_node(self, node, tf_variable):
        node_index = node.index if node is not None else -1
        if node_index not in self.nodesToVarsDict:
            self.nodesToVarsDict[node_index] = set()
        self.nodesToVarsDict[node_index].add(tf_variable)
        assert tf_variable not in self.varToNodesDict
        self.varToNodesDict[tf_variable] = node
        all_trainable_variables = set(tf.trainable_variables())
        is_trainable = tf_variable in all_trainable_variables
        if is_trainable:
            self.trainableVariables.append(tf_variable)
            is_residue_variable = "_residue_" in tf_variable.name
            op_name = self.network.get_assign_op_name(variable=tf_variable)
            new_value = tf.placeholder(name=op_name, dtype=GlobalConstants.DATA_TYPE)
            assign_op = tf.assign(ref=tf_variable, value=new_value)
            self.network.newValuesDict[op_name] = new_value
            self.network.assignOpsDict[op_name] = assign_op
            self.network.momentumStatesDict[tf_variable.name] = np.zeros(shape=tf_variable.shape)

    # Backward compatibility method
    def get_all_node_variables(self):
        for node in self.network.topologicalSortedNodes:
            for var in node.variablesSet:
                self.add_variable_to_node(node=node, tf_variable=var)
