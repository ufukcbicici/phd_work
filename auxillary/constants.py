from enum import Enum


class DatasetTypes(Enum):
    training = 0
    validation = 1
    test = 2


class ArgumentTypes(Enum):
    input = 0
    learnable_parameter = 1
    auxillary = 2
    label = 3
    decision_tolerance = 4
    output = 5