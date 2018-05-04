import tensorflow as tf


class VariableManager:
    def __init__(self, network):
        self.network = network
        self.trainableVariables = []
        self.varToNodesDict = {}
        self.nodesToVarsDict = {}

    def create_and_add_variable_to_node(self, node, name, initial_value, is_trainable=True):
        tf_variable = tf.Variable(initial_value=initial_value, name=name, trainable=is_trainable)
        node_index = node.index if node is not None else -1
        if node_index not in self.nodesToVarsDict:
            self.nodesToVarsDict[node_index] = set()
        self.nodesToVarsDict[node_index].add(tf_variable)
        return tf_variable

    def add_variable_to_node(self, node, tf_variable):
        node_index = node.index if node is not None else -1
        if node_index not in self.nodesToVarsDict:
            self.nodesToVarsDict[node_index] = set()
        self.nodesToVarsDict[node_index].add(tf_variable)

    def add_variables_to_node(self, node, tf_variables):
        for variable in tf_variables:
            self.add_variable_to_node(node=node, tf_variable=variable)

    def save_trainable_variables(self):
        self.trainableVariables = tf.trainable_variables()

    def trainable_variables(self):
        return self.trainableVariables


        # for var in vars:
        #     is_residue_variable = "_residue_" in var.name
        #     op_name = self.get_assign_op_name(variable=var)
        #     new_value = tf.placeholder(name=op_name, dtype=GlobalConstants.DATA_TYPE)
        #     assign_op = tf.assign(ref=var, value=new_value)
        #     self.newValuesDict[op_name] = new_value
        #     self.assignOpsDict[op_name] = assign_op
        #     self.momentumStatesDict[var.name] = np.zeros(shape=var.shape)
        #     for node in self.topologicalSortedNodes:
        #         if var in node.variablesSet:
        #             if var.name in self.varToNodesDict:
        #                 raise Exception("{0} is in the parameters already.".format(var.name))
        #             self.varToNodesDict[var.name] = node
        #     if not is_residue_variable and var.name not in self.varToNodesDict:
        #         raise Exception("{0} is not in the parameters!".format(var.name))
        #         # Add tensorboard ops
        #         # self.summaryFunc(network=self)
        #         # self.summaryWriter = tf.summary.FileWriter(GlobalConstants.SUMMARY_DIR + "//train")
