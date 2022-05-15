import numpy as np

from auxillary.parameters import DecayingParameter
from tf_2_cign.softmax_decay_algorithms.softmax_decay_algorithm import SoftmaxDecayAlgorithm


class StepWiseDecayAlgorithm(SoftmaxDecayAlgorithm):
    def __init__(self, decay_name, initial_value, decay_coefficient, decay_period, decay_min_limit):
        super().__init__()
        self.initialValue = initial_value
        self.decayCoefficient = decay_coefficient
        self.decayPeriod = decay_period
        self.decayMinLimit = decay_min_limit
        self.decayingParameter = DecayingParameter(name=decay_name,
                                                   value=self.initialValue,
                                                   decay=self.decayCoefficient,
                                                   decay_period=self.decayPeriod,
                                                   min_limit=self.decayMinLimit)

    def update(self, **kwargs):
        iteration = kwargs["iteration"]
        self.decayingParameter.update(iteration=iteration)

    def get_value(self):
        return self.decayingParameter.value

    def get_explanation(self, network, explanation, kv_rows):
        explanation = network.add_explanation(name_of_param="Temperature Decay Algorithm",
                                              value="StepWiseDecayAlgorithm",
                                              explanation=explanation, kv_rows=kv_rows)
        explanation = network.add_explanation(name_of_param="Temperature Initial",
                                              value=self.initialValue,
                                              explanation=explanation, kv_rows=kv_rows)
        explanation = network.add_explanation(name_of_param="Temperature Decay Coefficient",
                                              value=self.decayCoefficient,
                                              explanation=explanation, kv_rows=kv_rows)
        explanation = network.add_explanation(name_of_param="Temperature Decay Period",
                                              value=self.decayPeriod,
                                              explanation=explanation, kv_rows=kv_rows)
        explanation = network.add_explanation(name_of_param="Temperature Decay Min Limit",
                                              value=self.decayMinLimit,
                                              explanation=explanation, kv_rows=kv_rows)
        return explanation

        # explanation = ""
        # explanation += "Softmax Decay Initial:{0}\n".format(self.initialValue)
        # explanation += "Softmax Decay Coefficient:{0}\n".format(self.decayCoefficient)
        # explanation += "Softmax Decay Period:{0}\n".format(self.decayPeriod)
        # explanation += "Softmax Min Limit:{0}\n".format(self.decayMinLimit)
        # return explanation
