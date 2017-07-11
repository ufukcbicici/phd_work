from auxillary.dag_utilities import Dag
from auxillary.constants import OperationTypes, InputNames, InitType, ActivationType, PoolingType, TreeType, \
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

    def build_network(self):
        pass
