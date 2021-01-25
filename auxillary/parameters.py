import numpy as np


class Parameter:
    def __init__(self, name, value, default_value=0.0):
        self.name = name
        self.value = value
        self.initialValue = value
        self.logFileName = "hyperparameters.log"
        self.isActive = True
        self.defaultValue = default_value

    def reset(self):
        self.value = self.initialValue

    def enable(self):
        self.isActive = True

    def disable(self):
        self.isActive = False

    def get_value(self):
        if self.isActive:
            return self.value
        else:
            return self.defaultValue

    def set_value(self, value):
        self.value = value

    def update(self, iteration):
        pass

    def get_explanation(self):
        pass


class GaussianParameter(Parameter):
    def __init__(self, name, value, std, always_positive=False):
        Parameter.__init__(self, name=name, value=value)
        self.mean = value
        self.std = std
        self.alwaysPositive = always_positive

    def update(self, iteration):
        if not self.isActive:
            return
        self.value = np.random.normal(self.mean, self.std)
        if self.alwaysPositive:
            self.value = abs(self.value)


class DecayingParameter(Parameter):
    @staticmethod
    def from_training_program(name, training_program):
        value_dict = training_program.load_settings_for_property(property_name=name)
        decaying_parameter = \
            DecayingParameter(name=name,
                              value=value_dict["value"], decay=value_dict["decay"],
                              decay_period=value_dict["decayPeriod"], min_limit=value_dict["minLimit"],
                              epsilon_value=value_dict["epsilon_value"])
        return decaying_parameter

    # Change
    def __init__(self, name, value, decay, decay_period, min_limit=0.0, epsilon_value=None):
        Parameter.__init__(self, name=name, value=value)
        self.decay = decay
        self.decayPeriod = decay_period
        self.minLimit = min_limit
        self.epsilon_value = epsilon_value

    def update(self, iteration):
        if not self.isActive:
            return
        if iteration % self.decayPeriod == 0 and self.value > self.minLimit:
            self.value = max(self.decay * self.value, self.minLimit)
            # dbg_str = "Hyperparameter:{0} New value:{1}".format(self.name, self.value)
            # print(dbg_str)
            # BnnLogger.print_log(log_file_name=self.logFileName, log_string=dbg_str)
        if self.epsilon_value is not None:
            if self.value < self.epsilon_value:
                self.value = 0.0
        # if self.value < self.minLimit:
        #     self.value = self.minLimit

    def get_explanation(self):
        explanation = ""
        explanation += "Initial Value:{0}\n".format(self.value)
        explanation += "Decay Step:{0}\n".format(self.decayPeriod)
        explanation += "Decay Ratio:{0}\n".format(self.decay)
        explanation += "Minimum Limit:{0}\n".format(self.minLimit)
        return explanation


class DecayingParameterV2(Parameter):
    def __init__(self, name, value, decay, min_limit=0.0):
        Parameter.__init__(self, name=name, value=value)
        self.decay = decay
        self.minLimit = min_limit

    def update(self, iteration):
        if not self.isActive:
            return
        if self.value > self.minLimit:
            self.value = self.initialValue * (1.0 / (1.0 + float(iteration - 1.0)*self.decay))
        else:
            self.value = self.minLimit


class GrowingParameter(Parameter):
    # @staticmethod
    # def from_training_program(name, training_program):
    #     value_dict = training_program.load_settings_for_property(property_name=name)
    #     decaying_parameter = \
    #         DecayingParameter(name=name,
    #                           value=value_dict["value"], decay=value_dict["decay"],
    #                           decay_period=value_dict["growthPeriod"], min_limit=value_dict["minLimit"],
    #                           epsilon_value=value_dict["epsilon_value"])
    #     return decaying_parameter

    def __init__(self, name, value, growth, growth_period, max_limit):
        Parameter.__init__(self, name=name, value=value)
        self.growth = growth
        self.growthPeriod = growth_period
        self.maxLimit = max_limit

    def update(self, iteration):
        if not self.isActive:
            return
        if iteration % self.growthPeriod == 0 and self.value < self.maxLimit:
            self.value *= self.growth
            dbg_str = "Hyperparameter:{0} New value:{1}".format(self.name, self.value)
            print(dbg_str)


class DiscreteParameter(Parameter):
    def __init__(self, name, value, schedule):
        Parameter.__init__(self, name=name, value=value)
        self.schedule = schedule

    def update(self, iteration):
        if not self.isActive:
            return
        tpl = [v for v in self.schedule if v[0] == iteration]
        if tpl:
            self.value = tpl[0][1]
            self.schedule.remove(tpl[0])

    def get_explanation(self):
        explanation = "Iteration:{0} Value:{1}\n".format(0, self.value)
        for tpl in self.schedule:
            explanation += "Iteration:{0} Value:{1}\n".format(tpl[0], tpl[1])
        return explanation


class SinusoidalParameter(Parameter):
    def __init__(self, name, value, amplitude, period, phase=np.pi / 2.0):
        Parameter.__init__(self, name=name, value=value)
        self.amplitude = amplitude
        self.period = period
        self.phase = phase

    def update(self, iteration):
        if not self.isActive:
            return
        self.value = self.initialValue + self.amplitude * (
            np.sin(self.phase + 2.0 * np.pi * (iteration % self.period) / self.period) - 1.0)


class FixedParameter(Parameter):
    @staticmethod
    def from_training_program(name, training_program):
        value_dict = training_program.load_settings_for_property(property_name=name)
        fixed_parameter = FixedParameter(name=name, value=value_dict["value"])
        return fixed_parameter

    def __init__(self, name, value):
        Parameter.__init__(self, name=name, value=value)

    def update(self, iteration):
        pass

    def get_explanation(self):
        explanation = "Fixed Value:{0}\n".format(self.value)
        return explanation
