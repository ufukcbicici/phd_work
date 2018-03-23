from auxillary.constants import DatasetTypes
from auxillary.general_utility_funcs import UtilityFuncs


class SoftmaxCompresser:
    def __init__(self):
        pass

    @staticmethod
    def compress_network_softmax(network, sess, dataset):
        # Get all final feature vectors for all leaves, for the complete training set.
        posteriorsDict = {}
        dataset.set_current_data_set_type(dataset_type=DatasetTypes.training)
        while True:
            results = network.eval_network(sess=sess, dataset=dataset, use_masking=True)
            for leafNode in network.leafNodes:
                posterior_ref = network.get_variable_name(name="posterior_probs", node=leafNode)
                posterior_probs = results[posterior_ref]
                UtilityFuncs.concat_to_np_array_dict(dct=posteriorsDict, key=leafNode.index, array=posterior_probs)
            if dataset.isNewEpoch:
                break
        print("X")

