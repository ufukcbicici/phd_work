import numpy as np
from auxillary.constants import GlobalInputNames, LossType
from losses.sample_index_counter import SampleIndexCounter


class SgdOptimizer:
    def __init__(self, network, use_biased_gradient_estimates):
        self.network = network
        self.momentumStates = {}
        self.useBiasedGradientEstimates = use_biased_gradient_estimates

    def update(self):
        new_values_dict = {}
        batch_size = float(self.network.globalInputDrivers[GlobalInputNames.batch_size.value].value)
        for node in self.network.nodes.values():
            sample_index_counter_loss = node.losses[SampleIndexCounter.Name]
            num_of_samples = self.network.get_outputs_for_single_loss(loss_object=sample_index_counter_loss)[0]
            if num_of_samples == 0:
                continue
            for parameter in node.parametersDict.values():
                # Get the lr
                lr_hyper_param_name = parameter.get_property_name(property_=GlobalInputNames.lr.value)
                lr = self.network.globalInputDrivers[lr_hyper_param_name].value
                # Get the momentum decay rate
                momentum_hyper_param_name = parameter.get_property_name(property_=GlobalInputNames.momentum.value)
                momentum_decay = self.network.globalInputDrivers[momentum_hyper_param_name].value
                # Get the gradients wrt objective loss and wrt regularization loss
                objective_gradient = self.network.get_gradient_for_parameter(parameter_object=parameter,
                                                                             loss_type=LossType.objective)
                regularizer_gradient = self.network.get_gradient_for_parameter(parameter_object=parameter,
                                                                               loss_type=LossType.regularization)
                if parameter not in self.momentumStates:
                    self.momentumStates[parameter] = np.zeros(shape=parameter.valueArray.shape)
                momentum_state = self.momentumStates[parameter]
                total_gradient = lr * regularizer_gradient
                if self.useBiasedGradientEstimates:
                    total_gradient += lr * (batch_size / float(num_of_samples)) * objective_gradient
                momentum_state[:] *= momentum_decay
                momentum_state[:] -= total_gradient
                parameter.valueArray[:] += momentum_state
                new_values_dict[parameter.inputTensor] = parameter.valueArray
        return new_values_dict
