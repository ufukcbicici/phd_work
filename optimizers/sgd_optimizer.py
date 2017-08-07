from auxillary.constants import GlobalInputNames


class SgdOptimizer:
    def __init__(self, network):
        self.network = network
        self.momentumStates = {}

    def update(self):
        for node in self.network.nodes.values():
            for parameter in node.parametersDict.values():
                # Get the lr
                lr_hyper_param_name = parameter.get_property_name(property_=GlobalInputNames.lr.value)
                lr = self.network.globalInputDrivers[lr_hyper_param_name].value
                # Get the momentum
                momentum_hyper_param_name = parameter.get_property_name(property_=GlobalInputNames.momentum.value)
                momentum = self.network.globalInputDrivers[momentum_hyper_param_name].value

