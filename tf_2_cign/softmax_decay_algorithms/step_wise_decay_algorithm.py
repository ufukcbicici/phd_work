import numpy as np

from auxillary.parameters import DecayingParameter
from tf_2_cign.softmax_decay_algorithms.softmax_decay_algorithm import SoftmaxDecayAlgorithm


class StepWiseDecayAlgorithm(SoftmaxDecayAlgorithm):
    def __init__(self, decay_name, initial_value, decay_coefficient, decay_period, decay_min_limit):
        super().__init__()
        self.currTemperature = initial_value
        self.decayCoefficient = decay_coefficient
        self.decayPeriod = decay_period
        self.decayMinLimit = decay_min_limit
        self.decayingParameter = DecayingParameter(name=decay_name,
                                                   value=self.currTemperature,
                                                   decay=self.decayCoefficient,
                                                   decay_period=self.decayPeriod,
                                                   min_limit=self.decayMinLimit)

    def update_value(self, **kwargs):
        iteration = kwargs["iteration"]
        self.decayingParameter.update(iteration=iteration)
