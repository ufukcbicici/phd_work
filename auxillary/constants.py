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


class OperationTypes(Enum):
    input = "input"
    f_operator = "FOperator"
    h_operator = "HOperator"
    decision = "Decision"
    loss = "loss"


class OperationNames(Enum):
    data_input = "DataInput"


class ActivationType(Enum):
    relu = 0


class PoolingType(Enum):
    max = 0


class InitType(Enum):
    xavier = 0
