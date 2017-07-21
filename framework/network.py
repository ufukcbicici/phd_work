import tensorflow as tf
from auxillary.dag_utilities import Dag
from auxillary.constants import ChannelTypes, GlobalInputNames, InitType, ActivationType, PoolingType, TreeType, \
    ProblemType, ShrinkageRegularizers


class Network:
    def __init__(self, run_id, dataset, parameter_file, problem_type, loss_layer_init=InitType.xavier,
                 loss_activation=ActivationType.tanh, shrinkage_regularizer=ShrinkageRegularizers.l2):
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

    def build_network(self):
        pass

    def add_nodewise_input(self, producer_node, producer_channel, producer_channel_index, dest_node):
        pass

    def add_networkwise_input(self, name, channel, tensor_type):
        tensor = channel.add_operation(op=tf.placeholder(dtype=tensor_type, name=name))
        self.variablesToFeed.add(tensor)
        return tensor

    def apply_decision(self, node, channel, channel_index):
        pass