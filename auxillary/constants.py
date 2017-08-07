from enum import Enum


class DatasetTypes(Enum):
    training = 0
    validation = 1
    test = 2


class parameterTypes(Enum):
    input = 0
    learnable_parameter = 1
    auxillary = 2
    label = 3
    decision_tolerance = 4
    output = 5


class ChannelTypes(Enum):
    data_input = "DataInput"
    label_input = "LabelInput"
    indices_input = "IndicesInput"
    f_operator = "FOperator"
    h_operator = "HOperator"
    branching_activation = "BranchingActivation"
    branching_probabilities = "BranchingProbabilities"
    branching_masks_unified = "BranchingMasksUnified"
    branching_masks_sliced = "BranchingMasksSliced"
    decision = "Decision"
    pre_loss = "PreLoss"
    loss = "Loss"
    total_loss = "TotalLoss"
    gradient = "Gradient"
    evaluation = "Evaluation"
    constant = "Constant"
    sample_indices = "SampleIndices"
    parameter_update = "ParameterUpdate"


class GlobalInputNames(Enum):
    global_scope = "GlobalScope"
    regularizer_strength = "RegularizerStrength"
    branching_prob_threshold = "BranchingProbThreshold"
    batch_size = "BatchSize"
    wd = "wd"
    lr = "lr"
    lr_initial = "lr_initial"
    momentum = "momentum"
    weight_update_interval = "weight_update_interval"
    lr_update_interval = "lr_update_interval"
    lr_decay_ratio = "lr_decay_ratio"
    parameter_update = "parameter_update"


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