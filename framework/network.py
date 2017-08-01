import tensorflow as tf
from auxillary.dag_utilities import Dag
from auxillary.constants import ChannelTypes, GlobalInputNames, InitType, ActivationType, PoolingType, TreeType, \
    ProblemType, ShrinkageRegularizers
from framework.network_channel import NetworkChannel


class Network:
    def __init__(self, run_id, dataset, parameter_file, problem_type,
                 loss_layer_init=InitType.xavier,
                 loss_activation=ActivationType.tanh,
                 activation_init=InitType.xavier,
                 shrinkage_regularizer=ShrinkageRegularizers.l2):
        self.nodes = {}
        self.leafNodes = []
        self.runId = run_id
        self.dataset = dataset
        self.parameterFile = parameter_file
        self.dag = Dag()
        self.topologicalSortedNodes = []
        self.problemType = problem_type
        self.lossLayerInit = loss_layer_init
        self.lossLayerActivation = loss_activation
        self.variablesToFeed = set()
        self.shrinkageRegularizer = shrinkage_regularizer
        self.indicatorText = None
        self.activationInit = activation_init
        self.accumulationNode = None
        # Tensors to be used in training
        self.trainingTensorsList = []
        # Tensors to be used in evaluation
        self.evaluationTensorsList = []

    def build_network(self):
        pass

    def add_nodewise_input(self, producer_node, producer_channel, producer_channel_index, dest_node):
        pass

    def create_global_inputs(self):
        pass

    def get_accumulation_node(self):
        if self.accumulationNode is None:
            candidates = []
            for node in self.nodes.values():
                if node.isAccumulation:
                    candidates.append(node)
            if len(candidates) != 1:
                raise Exception("There must be exactly one accumulation node in the network.")
            self.accumulationNode = candidates[0]
        return self.accumulationNode

    # All networkwise inputs are constant, and they will be placed into the accumulation node.
    def add_networkwise_input(self, name, tensor_type):
        tensor_names = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        for tensor_name in tensor_names:
            if name in tensor_name:
                raise Exception("Name {0} is already used in tensor name {1}".format(name, tensor_name))
        with NetworkChannel(parent_node=None,
                            parent_node_channel=ChannelTypes.constant) as constant_channel:
            tensor = constant_channel.add_operation(op=tf.placeholder(dtype=tensor_type, name=name))
            self.variablesToFeed.add(tensor)
        return tensor

    def get_networkwise_input(self, name):
        accumulation_node = self.get_accumulation_node()
        candidates = []
        for output in accumulation_node.outputs.values():
            if name in output.tensor.name:
                candidates.append(output.tensor)
        if len(candidates) != 1:
            raise Exception("There must be 1 tensor returned from get_networkwise_input")
        tensor = candidates[0]
        return tensor



