class Node:
    def __init__(self, index, depth, is_root, is_leaf):
        self.index = index
        self.depth = depth
        self.isRoot = is_root
        self.isLeaf = is_leaf
        self.variablesList = []
        self.fOpsList = []
        self.hOpsList = []
        self.lossList = []
        self.labelTensor = None
        self.maskTensorsDict = {}
        self.evalDict = {}
        # Indexed by the nodes producing them
        self.activationsDict = {}