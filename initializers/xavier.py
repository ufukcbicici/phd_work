import numpy as np


class Xavier:
    """Returns an initializer performing "Xavier" initialization for weights.
    This initializer is designed to keep the scale of gradients roughly the same
    in all layers.
    By default, `rnd_type` is ``'uniform'`` and `factor_type` is ``'avg'``,
    the initializer fills the weights with random numbers in the range
    of :math:`[-c, c]`, where :math:`c = \\sqrt{\\frac{3.}{0.5 * (n_{in} + n_{out})}}`.
    :math:`n_{in}` is the number of neurons feeding into weights, and :math:`n_{out}` is
    the number of neurons the result is fed to.
    If `rnd_type` is ``'uniform'`` and `factor_type` is ``'in'``,
    the :math:`c = \\sqrt{\\frac{3.}{n_{in}}}`.
    Similarly when `factor_type` is ``'out'``, the :math:`c = \\sqrt{\\frac{3.}{n_{out}}}`.
    If `rnd_type` is ``'gaussian'`` and `factor_type` is ``'avg'``,
    the initializer fills the weights with numbers from normal distribution with
    a standard deviation of :math:`\\sqrt{\\frac{3.}{0.5 * (n_{in} + n_{out})}}`.
    Parameters
    ----------
    rnd_type: str, optional
        Random generator type, can be ``'gaussian'`` or ``'uniform'``.
    factor_type: str, optional
        Can be ``'avg'``, ``'in'``, or ``'out'``.
    magnitude: float, optional
        Scale of random number.
    """

    def __init__(self, rnd_type="uniform", factor_type="avg", magnitude=3):
        self.rnd_type = rnd_type
        self.factor_type = factor_type
        self.magnitude = float(magnitude)

    def init_weight(self, arr):
        shape = arr.shape
        hw_scale = 1.
        if len(shape) < 2:
            return np.zeros(shape=arr.shape)
        if len(shape) > 2:
            hw_scale = np.prod(shape[2:])
        fan_in, fan_out = shape[1] * hw_scale, shape[0] * hw_scale
        factor = 1.
        if self.factor_type == "avg":
            factor = (fan_in + fan_out) / 2.0
        elif self.factor_type == "in":
            factor = fan_in
        elif self.factor_type == "out":
            factor = fan_out
        else:
            raise ValueError("Incorrect factor type")
        scale = np.sqrt(self.magnitude / factor)
        if self.rnd_type == "uniform":
            return np.random.uniform(low=-scale, high=scale, size=shape)
        elif self.rnd_type == "gaussian":
            return np.random.normal(loc=0, scale=scale, size=shape)
        else:
            raise ValueError("Unknown random type")