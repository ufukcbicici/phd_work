from simple_tf.global_params import Pipeline


class Node:
    def __init__(self, index, depth, is_root, is_leaf):
        self.index = index
        self.depth = depth
        self.isRoot = is_root
        self.isLeaf = is_leaf
        self.variablesSet = set()
        self.fOpsList = {}
        self.hOpsList = {}
        self.lossList = {}
        self.labelTensors = {}
        self.indicesTensors = {}
        self.isOpenIndicatorTensors = {}
        self.maskTensors = {}
        self.evalDict = {}
        self.activationsDict = {}
        self.proxyLossInputDicts = {}
        for pipeline in {Pipeline.thresholded, Pipeline.non_thresholded}:
            self.fOpsList[pipeline] = []
            self.hOpsList[pipeline] = []
            self.lossList[pipeline] = []
            self.labelTensors[pipeline] = None
            self.indicesTensors[pipeline] = None
            self.isOpenIndicatorTensors[pipeline] = None
            self.maskTensors[pipeline] = None
            self.activationsDict[pipeline] = {}
        # self.maskTensorsWithThresholdDict = {}
        # self.maskTensorsWithoutThresholdDict = {}

        # self.parentNonThresholdMaskVector = None
        # Indexed by the nodes producing them
