import numpy as np
from tf_2_cign.cigt.data_classes.multipath_routing_info2 import MultipathCombinationInfo2


# class HighEntropyErrorAnalysis:
#     def __init__(self, model_id, model_loader):
#         self.modelId = model_id
#         self.modelLoader = model_loader
#         self.model, self.dataset = self.modelLoader.get_model(model_id=self.modelId)
#         self.pathCounts = list(self.model.pathCounts)
#         self.multiPathInfoObject = self.load_multipath_info()
#
#     def load_multipath_info(self):
#         multipath_info_object = MultipathCombinationInfo2(batch_size=self.model.batchSize,
#                                                           path_counts=self.pathCounts)
#         multipath_info_object.generate_routing_info(cigt=self.model,
#                                                     dataset=self.dataset.testDataTf)
#         multipath_info_object.assert_routing_validity(cigt=self.model)
#         multipath_info_object.assess_accuracy()
#         return multipath_info_object
#
#     def run(self):
#         routing_decisions_arr = np.zeros(shape=(len(indices), sum(cigt.pathCounts[1:])), dtype=np.int32)
#         curr_index = 0
#         for block_id in range(len(cigt.pathCounts) - 1):
#             routing_decisions_so_far = routing_decisions_arr[:, :curr_index]
#             index_arrays = [routing_decisions_so_far[:, col] for col in range(routing_decisions_so_far.shape[1])]
#             index_arrays.append(indices)
#             routing_probabilities_for_block = \
#                 self.past_decisions_routing_probabilities_list[block_id][index_arrays]
#             decision_array = np.zeros(shape=routing_probabilities_for_block.shape,
#                                       dtype=routing_decisions_so_far.dtype)
#             decision_array[np.arange(routing_probabilities_for_block.shape[0]),
#                            np.argmax(routing_probabilities_for_block, axis=1)] = 1
#             routing_decisions_arr[:, curr_index:curr_index + decision_array.shape[1]] = decision_array
#             curr_index += cigt.pathCounts[block_id + 1]
#         idx_arr = np.concatenate([routing_decisions_arr, np.expand_dims(indices, axis=1)], axis=1)
#         idx_arr = [idx_arr[:, col] for col in range(idx_arr.shape[1])]
#         validity_vec = self.past_decisions_validity_array[idx_arr]
#         accuracy = np.mean(validity_vec)
#         print("Accuracy:{0}".format(accuracy))