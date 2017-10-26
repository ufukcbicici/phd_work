class Node:
    def __init__(self, index, depth, is_root, is_leaf):
        self.index = index
        self.depth = depth
        self.isRoot = is_root
        self.isLeaf = is_leaf
        self.variablesSet = set()
        self.fOpsList = []
        self.hOpsList = []
        self.lossList = []
        self.labelTensor = None
        self.indicesTensor = None
        self.isOpenIndicatorTensor = None
        self.maskTensors = {}
        self.evalDict = {}
        self.parentNonThresholdMaskVector = None
        # Indexed by the nodes producing them
        self.activationsDict = {}
        self.proxyLossInputDicts = {}