import numpy as np

from auxillary.parameters import DecayingParameter
from tf_2_cign.softmax_decay_algorithms.softmax_decay_algorithm import SoftmaxDecayAlgorithm


class HyperbolicDecayAlgorithm(SoftmaxDecayAlgorithm):
    def __init__(self, decay_name, initial_value, exponent, decay_min_limit):
        super().__init__()
        self.decayName = decay_name
        self.initialValue = initial_value
        self.exponent = exponent
        self.decayMinLimit = decay_min_limit
        self.currValue = initial_value

    def update(self, **kwargs):
        iteration = kwargs["iteration"]
        self.currValue = max(self.initialValue / (float(iteration + 1)**self.exponent), self.decayMinLimit)

    def get_value(self):
        return self.currValue

    def get_explanation(self, network, explanation, kv_rows):
        explanation = network.add_explanation(name_of_param="Temperature Decay Algorithm",
                                              value="HyperbolicDecayAlgorithm",
                                              explanation=explanation, kv_rows=kv_rows)
        explanation = network.add_explanation(name_of_param="Temperature Initial",
                                              value=self.initialValue,
                                              explanation=explanation, kv_rows=kv_rows)
        explanation = network.add_explanation(name_of_param="Temperature Exponent",
                                              value=self.exponent,
                                              explanation=explanation, kv_rows=kv_rows)
        explanation = network.add_explanation(name_of_param="Temperature Decay Min Limit",
                                              value=self.decayMinLimit,
                                              explanation=explanation, kv_rows=kv_rows)
        return explanation

    # def get_explanation(self):
    #     explanation = ""
    #     explanation += "Hyperbolic Decay Initial:{0}\n".format(self.initialValue)
    #     explanation += "Hyperbolic Decay Exponent:{0}\n".format(self.exponent)
    #     explanation += "Hyperbolic Min Limit:{0}\n".format(self.decayMinLimit)
    #     return explanation
