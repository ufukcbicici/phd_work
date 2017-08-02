class GenericLoss:
    def __init__(self, parent_node, name):
        self.parentNode = parent_node
        self.lossOutputs = None
        self.evalOutputs = None
        self.isFinalized = False
        self.name = name
        self.lossIndex = None
        self.evalIndex = None

    def build_training_network(self):
        pass

    def build_evaluation_network(self):
        pass

    def finalize(self):
        if self.name in self.parentNode.losses:
            raise Exception("Loss {0} is already in node {1}".format(self.name, self.parentNode))
        self.parentNode.losses[self.name] = self
        self.isFinalized = True
