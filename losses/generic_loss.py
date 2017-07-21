class GenericLoss:
    def __init__(self, parent_node):
        self.parentNode = parent_node
        self.lossOutput = None
        self.evalOutput = None
        self.isFinalized = False

    def build_training_network(self):
        pass

    def build_evaluation_network(self):
        pass

    def finalize(self):
        self.isFinalized = True
