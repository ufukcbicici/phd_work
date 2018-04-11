from algorithms.softmax_compresser import SoftmaxCompresser
from auxillary.general_utility_funcs import UtilityFuncs


class NodeMock:
    def __init__(self, index):
        self.index = index

# "C:\Users\t67rt\Desktop\phd_work\phd_work\simple_tf\parameters.npz"
node_3_dict = UtilityFuncs.load_npz(
    file_name="C://Users//t67rt//Desktop//phd_work//phd_work//simple_tf//npz_node_3_distillation")
node_4_dict = UtilityFuncs.load_npz(
    file_name="C://Users//t67rt//Desktop//phd_work//phd_work//simple_tf//npz_node_4_distillation")

mock_node = NodeMock(index=4)
SoftmaxCompresser.assert_prob_correctness(softmax_weights=node_4_dict["softmax_weights"],
                                          softmax_biases=node_4_dict["softmax_biases"],
                                          features=node_4_dict["features"],
                                          logits=node_4_dict["logits"],
                                          probs=node_4_dict["probs"],
                                          leaf_node=mock_node)