# import tensorflow as tf
# import numpy as np
#
# from simple_tf.cign.cign_mixture_of_experts import CignMixtureOfExperts
# from simple_tf.global_params import GlobalConstants
#
#
# class CignMixtureOfExpertsLogitEnsemble(CignMixtureOfExperts):
#     def __init__(self, node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
#                  dataset):
#         super().__init__(node_build_funcs, grad_func, hyperparameter_func, residue_func, summary_func, degree_list,
#                          dataset)
#         GlobalConstants.USE_UNIFIED_BATCH_NORM = False
#
#     def apply_loss(self, node, final_feature, softmax_weights, softmax_biases):
#         node.residueOutputTensor = final_feature
#         node.finalFeatures = final_feature
#         node.evalDict[self.get_variable_name(name="final_feature_final", node=node)] = final_feature
#         node.evalDict[self.get_variable_name(name="final_feature_mag", node=node)] = tf.nn.l2_loss(final_feature)
#         routing_probs = self.get_node_routing_probabilities(node=node)
#         node.evalDict[self.get_variable_name(name="routing_probs", node=node)] = routing_probs
#         logits = tf.matmul(final_feature, softmax_weights) + softmax_biases
#         node.evalDict[self.get_variable_name(name="logits", node=node)] = logits
#         cross_entropy_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node.labelTensor,
#                                                                                    logits=logits)
#         weighted_cross_entropy_loss_tensor = cross_entropy_loss_tensor * routing_probs
#         loss = tf.reduce_mean(weighted_cross_entropy_loss_tensor)
#         node.lossList.append(loss)
#         return final_feature, logits
