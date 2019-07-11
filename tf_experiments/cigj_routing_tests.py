import tensorflow as tf
import numpy as np
from auxillary.constants import DatasetTypes
from auxillary.general_utility_funcs import UtilityFuncs
from data_handling.data_set import DataSet
from data_handling.fashion_mnist import FashionMnistDataSet
from simple_tf.cigj.jungle import Jungle
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.fashion_net.fashion_net_cigj import FashionNetCigj
from simple_tf.uncategorized.global_params import GlobalConstants


class CigjTesting:
    @staticmethod
    def transfer_single_path_parameters_with_shape(sess,
                                                   jungle,
                                                   jungle_parameters,
                                                   single_path_jungle,
                                                   single_path_parameters,
                                                   single_path_placehoders,
                                                   single_path_assignment_ops,
                                                   path):
        parameter_pairs = []
        extended_path = [0]
        extended_path.extend(path)
        source_nodes = []
        for source_node in jungle.topologicalSortedNodes:
            if source_node.nodeType == NodeType.h_node:
                continue
            parameters_source = []
            parameters_destination = []
            # Get all variables of a f_node
            sibling_order_index = jungle.get_node_sibling_index(node=source_node)
            depth = source_node.depth
            if sibling_order_index != extended_path[depth]:
                continue
            source_nodes.append(source_node)
            parameters_source.extend([param for param in jungle_parameters
                                      if "Node{0}".format(source_node.index) in param.name])
            destination_nodes = [_node for _node in single_path_jungle.nodes.values()
                                 if _node.nodeType != NodeType.h_node and _node.depth == depth]
            assert len(destination_nodes) == 1
            destination_node = destination_nodes[0]
            parameters_destination.extend([param for param in single_path_parameters
                                           if "Node{0}".format(destination_node.index) in param.name])
            assert len(parameters_source) == len(parameters_destination)
            # Match parameters in two nodes
            for s_param in parameters_source:
                s_suffix = s_param.name[len("jungle/Node{0}".format(source_node.index)):]
                d_candidates = [d_param for d_param in parameters_destination
                                if s_suffix ==
                                d_param.name[len("single_path_jungle/Node{0}".format(destination_node.index)):]]
                assert len(d_candidates) == 1
                d_param = d_candidates[0]
                parameter_pairs.append((s_param, d_param))
        # Check if the mapping has been correctly done.
        # Check consistency of the shapes
        assert all([tpl[0].get_shape().as_list() == tpl[1].get_shape().as_list() for tpl in parameter_pairs])
        # Check that mapping is one-to-one
        assert len(set([tpl[1] for tpl in parameter_pairs])) == len(parameter_pairs)
        # Get the source parameter values
        source_param_values = sess.run([tpl[0] for tpl in parameter_pairs])
        dest_param_values = sess.run([tpl[1] for tpl in parameter_pairs])
        assert not all([np.array_equal(sa, dest_param_values[_i]) for _i, sa in enumerate(source_param_values)])
        # Assign the source values to destination parameters
        feed_dict = {single_path_placehoders[tpl[1].name]: source_param_values[_i] for _i, tpl in
                     enumerate(parameter_pairs)}
        sess.run([assign_op for assign_op in single_path_assignment_ops.values()], feed_dict=feed_dict)
        dest_param_values2 = sess.run([tpl[1] for tpl in parameter_pairs])
        assert all([np.array_equal(sa, dest_param_values2[_i]) for _i, sa in enumerate(source_param_values)])
        return source_nodes

    @staticmethod
    def test():
        # Create the dataset and the model
        dataset = FashionMnistDataSet(validation_sample_count=0, load_validation_from=None)
        dataset.set_current_data_set_type(dataset_type=DatasetTypes.training, batch_size=GlobalConstants.BATCH_SIZE)
        node_build_funcs = [FashionNetCigj.f_conv_layer_func,
                            FashionNetCigj.f_conv_layer_func,
                            FashionNetCigj.f_conv_layer_func,
                            FashionNetCigj.f_fc_layer_func,
                            FashionNetCigj.f_leaf_func]
        with tf.name_scope('jungle'):
            jungle = Jungle(
                node_build_funcs=node_build_funcs,
                h_funcs=[FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func],
                grad_func=None,
                hyperparameter_func=FashionNetCigj.threshold_calculator_func,
                residue_func=None, summary_func=None,
                degree_list=GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST, dataset=dataset)
        jungle_parameters = set(tf.trainable_variables())
        # Build a single root-to-leaf model
        with tf.name_scope('single_path_jungle'):
            single_path_jungle = Jungle(
                node_build_funcs=node_build_funcs,
                h_funcs=[FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func],
                grad_func=None,
                hyperparameter_func=FashionNetCigj.threshold_calculator_func,
                residue_func=None, summary_func=None,
                degree_list=[1] * len(GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST), dataset=dataset)
        single_path_parameters = set([var for var in tf.trainable_variables() if var not in jungle_parameters])
        single_path_placehoders = {param.name: tf.placeholder(dtype=tf.float32) for param in single_path_parameters}
        single_path_assignment_ops = {param.name: tf.assign(param, single_path_placehoders[param.name]) for param in
                                      single_path_parameters}
        shape_set = set(tuple(v.get_shape().as_list()) for v in single_path_parameters)
        assert len(shape_set) == len(single_path_parameters)
        # Create all root-to-leaf model combinations
        list_of_indices = []
        for degree in GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST:
            list_of_indices.append([i for i in range(degree)])
        cartesian_product = UtilityFuncs.get_cartesian_product(list_of_lists=list_of_indices)
        # Run the Jungle for a single pass
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        minibatch = dataset.get_next_batch(batch_size=GlobalConstants.EVAL_BATCH_SIZE)
        results, _ = jungle.eval_minibatch(sess=sess, minibatch=minibatch, use_masking=True)
        # Extract each sample's route in the CIGJ
        ordered_h_nodes = [node for node in jungle.topologicalSortedNodes if node.nodeType == NodeType.h_node]
        sample_paths_dict = {sample_index: [] for sample_index in range(GlobalConstants.EVAL_BATCH_SIZE)}
        for h_node in ordered_h_nodes:
            indices_tensor = results[UtilityFuncs.get_variable_name(name="indices_tensor", node=h_node)]
            for sample_index in range(GlobalConstants.EVAL_BATCH_SIZE):
                sample_paths_dict[sample_index].append(indices_tensor[sample_index])
        paths_to_samples_dict = {tuple(path): [] for path in set([tuple(v) for v in sample_paths_dict.values()])}
        for sample_index, path in sample_paths_dict.items():
            paths_to_samples_dict[tuple(path)].append(sample_index)
        print("X")
        # For every path combination, extract the corresponding data and label subsets and run on the single path CNN.
        # Compare the results of each output with the CIGJ.
        largest_relative_error = -1
        largest_deviated_samples = None
        for path, samples in paths_to_samples_dict.items():
            print("Path:{0}".format(path))
            sorted_samples = sorted(samples)
            # samples_subset = minibatch.samples[sorted_samples]
            # labels_subset = minibatch.labels[sorted_samples]
            subset_minibatch = DataSet.MiniBatch(minibatch.samples[sorted_samples], minibatch.labels[sorted_samples],
                                                 minibatch.indices[sorted_samples],
                                                 minibatch.one_hot_labels[sorted_samples],
                                                 None, None, None)
            source_nodes = \
                CigjTesting.transfer_single_path_parameters_with_shape(sess=sess,
                                                                       jungle=jungle,
                                                                       jungle_parameters=jungle_parameters,
                                                                       single_path_jungle=single_path_jungle,
                                                                       single_path_parameters=single_path_parameters,
                                                                       single_path_placehoders=single_path_placehoders,
                                                                       single_path_assignment_ops=single_path_assignment_ops,
                                                                       path=path)
            single_path_results, _ = single_path_jungle.eval_minibatch(sess=sess, minibatch=subset_minibatch,
                                                                       use_masking=True)
            # results_list = []
            # for _ in range(1000):
            #     p, q = single_path_jungle.eval_minibatch(sess=sess, minibatch=subset_minibatch, use_masking=True)
            #     results_list.append(p)
            # single_path_results2, _ = single_path_jungle.eval_minibatch(sess=sess, minibatch=subset_minibatch,
            #                                                            use_masking=True)
            # single_path_results3, _ = single_path_jungle.eval_minibatch(sess=sess, minibatch=subset_minibatch,
            #                                                            use_masking=True)
            depthwise_results_equal = {}
            depthwise_results_allclose = {}
            for d in range(len(GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST)):
                for batch_index in sorted_samples:
                    depthwise_results_equal[(d, batch_index)] = False
                    depthwise_results_allclose[(d, batch_index)] = False
            # Get corresponding jungle results for the current path
            for source_node in source_nodes:
                # Get corresponding jungle node F output
                source_F_output = results[UtilityFuncs.get_variable_name(name="F_output", node=source_node)]
                source_F_condition_indices = results[UtilityFuncs.get_variable_name(name="condition_indices",
                                                                                    node=source_node)]
                destination_nodes = [_node for _node in single_path_jungle.nodes.values()
                                     if _node.nodeType != NodeType.h_node and _node.depth == source_node.depth]
                assert len(destination_nodes) == 1
                destination_node = destination_nodes[0]
                destination_F_output = single_path_results[UtilityFuncs.get_variable_name(name="F_output",
                                                                                          node=destination_node)]
                for d_condition_index, d_batch_index in enumerate(sorted_samples):
                    d_F = destination_F_output[d_condition_index]
                    for s_condition_index, s_batch_index in enumerate(source_F_condition_indices):
                        if s_batch_index == d_batch_index:
                            s_F = source_F_output[s_condition_index]
                            depthwise_results_equal[(source_node.depth, d_batch_index)] = np.array_equal(d_F, s_F)
                            depthwise_results_allclose[(source_node.depth, d_batch_index)] = np.allclose(d_F, s_F,
                                                                                                         rtol=1e-03)
                            print("Output={0} Equal:{1} Allclose:{2}".
                                  format((source_node.depth, d_batch_index),
                                         depthwise_results_equal[(source_node.depth, d_batch_index)],
                                         depthwise_results_allclose[(source_node.depth, d_batch_index)]))
                            # if depthwise_results_equal[(source_node.depth, d_batch_index)] is False:
                            #     print("X")
                            if depthwise_results_allclose[(source_node.depth, d_batch_index)] is False:
                                max_abs_diff = np.max(np.abs(d_F - s_F))
                                max_diff_index = np.unravel_index(np.argmax(np.abs(d_F - s_F)), d_F.shape)
                                max_relative_diff = np.abs(s_F[max_diff_index] - d_F[max_diff_index]) / \
                                                        np.abs(d_F[max_diff_index])
                                if max_relative_diff > largest_relative_error:
                                    largest_relative_error = max_relative_diff
                                    largest_deviated_samples = [s_F[max_diff_index], d_F[max_diff_index]]
                                    print("New max relative error:{0} s_F:{1} d_F:{2}".format(largest_relative_error,
                                                                                              s_F[max_diff_index],
                                                                                              d_F[max_diff_index]))
                                # print("{0} max_relative_diff:{1} max_abs_diff:{2} s_F:{3} d_F:{4} max_diff_index:{5}".
                                #       format((source_node.depth, d_batch_index), max_relative_diff, max_abs_diff,
                                #              s_F[max_diff_index], d_F[max_diff_index], max_diff_index))

            # assert all(depthwise_results_equal.values())
            # assert all(depthwise_results_allclose.values())
        print("X")


CigjTesting.test()
