import tensorflow as tf
from auxillary.constants import DatasetTypes
from auxillary.general_utility_funcs import UtilityFuncs
from data_handling.fashion_mnist import FashionMnistDataSet
from simple_tf.cigj.jungle import Jungle
from simple_tf.cigj.jungle_node import NodeType
from simple_tf.fashion_net.fashion_net_cigj import FashionNetCigj
from simple_tf.global_params import GlobalConstants


class CigjTesting:
    @staticmethod
    def transfer_single_path_parameters_with_shape(jungle, jungle_parameters,
                                                   single_path_jungle, single_path_parameters, path):
        parameter_pairs = []
        extended_path = [0]
        extended_path.extend(path)
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
            print("X")
        # Check if the mapping has been correctly done.
        # Check consistency of the shapes
        assert all([tpl[0].get_shape().as_list() == tpl[1].get_shape().as_list() for tpl in parameter_pairs])
        # Check that mapping is one-to-one
        assert len(set([tpl[1] for tpl in parameter_pairs])) == len(parameter_pairs)

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
                threshold_func=FashionNetCigj.threshold_calculator_func,
                residue_func=None, summary_func=None,
                degree_list=GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST, dataset=dataset)
        jungle_parameters = set(tf.trainable_variables())
        # Build a single root-to-leaf model
        with tf.name_scope('single_path_jungle'):
            single_path_jungle = Jungle(
                node_build_funcs=node_build_funcs,
                h_funcs=[FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func, FashionNetCigj.h_func],
                grad_func=None,
                threshold_func=FashionNetCigj.threshold_calculator_func,
                residue_func=None, summary_func=None,
                degree_list=[1] * len(GlobalConstants.CIGJ_FASHION_NET_DEGREE_LIST), dataset=dataset)
        single_path_parameters = set([var for var in tf.trainable_variables() if var not in jungle_parameters])
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
        for path, samples in paths_to_samples_dict.items():
            sorted_samples = sorted(samples)
            samples_subset = minibatch.samples[sorted_samples]
            labels_subset = minibatch.labels[sorted_samples]
            CigjTesting.transfer_single_path_parameters_with_shape(jungle=jungle,
                                                                   jungle_parameters=jungle_parameters,
                                                                   single_path_jungle=single_path_jungle,
                                                                   single_path_parameters=single_path_parameters,
                                                                   path=path)

            print("X")


CigjTesting.test()
