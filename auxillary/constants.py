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


class ChannelTypes(Enum):
    data_input = "DataInput"
    label_input = "LabelInput"
    f_operator = "FOperator"
    h_operator = "HOperator"
    ancestor_activation = "AncestorActivation"
    decision = "Decision"
    pre_loss = "PreLoss"
    loss = "Loss"
    gradient = "Gradient"
    evaluation = "Evaluation"
    constant = "Constant"


class GlobalInputNames(Enum):
    data_input = "DataInput"
    label_input = "LabelInput"
    regularizer_strength = "RegularizerStrength"


class ActivationType(Enum):
    relu = 0
    tanh = 1
    no_activation = 2


class PoolingType(Enum):
    max = 0


class InitType(Enum):
    xavier = 0


class TreeType(Enum):
    hard = 0
    soft = 1


class ProblemType(Enum):
    classification = 0
    regression = 1


class SpecialNodeTypes(Enum):
    data_provider = 0


class SpecialChannelIndices(Enum):
    data_provider_index = 0


class TrainingHyperParameters:
    wd = 0
    lr = 1


class ShrinkageRegularizers:
    l2 = 0