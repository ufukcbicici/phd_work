from auxillary.dag_utilities import Dag
from auxillary.constants import ChannelTypes, GlobalInputNames, InitType, ActivationType, PoolingType, TreeType, \
    ProblemType


class Network:
    def __init__(self, run_id, dataset, parameter_file, problem_type, loss_layer_init=InitType.xavier,
                 loss_activation=ActivationType.tanh):
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
        self.variablesToFeed = {}
        self.indicatorText = None

    def build_network(self):
        pass

    def add_nodewise_input(self, producer_node, producer_channel, producer_channel_index, dest_node):
        pass

    def add_networkwise_inputs(self):
        pass

    def apply_decision(self, node, channel, channel_index):
        pass
