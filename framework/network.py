from auxillary.dag_utilities import Dag


class Network:
    def __init__(self, run_id, dataset, parameter_file):
        self.nodes = {}
        self.leafNodes = []
        self.runId = run_id
        self.dataset = dataset
        self.parameterFile = parameter_file
        self.dag = Dag()
        self.topologicalSortedNodes = []

    def build_network(self):
        pass
