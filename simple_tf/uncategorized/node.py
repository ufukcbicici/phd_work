import tensorflow as tf


class Node:
    def __init__(self, index, depth, is_root, is_leaf):
        self.index = index
        self.depth = depth
        self.isRoot = is_root
        self.isLeaf = is_leaf
        self.fOpsList = []
        self.hOpsList = []
        self.lossList = []
        self.infoGainLoss = None
        self.labelTensor = None
        self.labelMappingTensor = None
        self.compressedLabelsTensor = None
        self.oneHotLabelTensor = None
        self.indicesTensor = None
        self.isOpenIndicatorTensor = None
        self.maskTensors = {}
        self.masksWithoutThreshold = {}
        self.filteredMask = None
        self.evalDict = {}
        self.parentNonThresholdMaskVector = None
        self.probabilityThreshold = None
        self.softmaxDecay = None
        self.probThresholdCalculator = None
        self.softmaxDecayCalculator = None
        self.finalFeatures = None
        self.residueOutputTensor = None
        self.weightDecayModifier = 1.0
        self.infoGainBalanceCoefficient = None
        self.p_n_given_x = None
        self.leafCountUnderThisNode = 0
        self.macCost = 0
        self.opMacCostsDict = {}
        self.batchIndicesTensor = None
        # Indexed by the nodes producing them
        self.activationsDict = {}
        self.proxyLossInputDicts = {}
