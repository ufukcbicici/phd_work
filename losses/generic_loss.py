class GenericLoss:
    def __init__(self, parent_node, loss_type, is_differentiable):
        self.parentNode = parent_node
        self.lossOutputs = None
        self.auxOutputs = None
        self.evalOutputs = None
        self.isFinalized = False
        self.lossType = loss_type
        self.isDifferentiable = is_differentiable

    def build_training_network(self):
        pass

    def build_evaluation_network(self):
        pass

    def finalize(self):
        name = self.get_name()
        if name in self.parentNode.losses:
            raise Exception("Loss {0} is already in node {1}".format(name, self.parentNode))
        self.parentNode.losses[name] = self
        self.isFinalized = True

    def get_name(self):
        pass
