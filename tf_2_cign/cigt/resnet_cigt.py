from tf_2_cign.cigt.cigt import Cigt
import json

from tf_2_cign.cigt.custom_layers.resnet_layers.resnet_cigt_inner_block import ResnetCigtInnerBlock
from tf_2_cign.cigt.custom_layers.resnet_layers.resnet_cigt_leaf_block import ResnetCigtLeafBlock
from tf_2_cign.cigt.custom_layers.resnet_layers.resnet_cigt_root_block import ResnetCigtRootBlock
from tf_2_cign.utilities.resnet_cigt_constants import ResnetCigtConstants


class ResnetCigt(Cigt):
    def __init__(self, run_id, model_definition, *args, **kwargs):
        self.resnetConfigList = ResnetCigtConstants.resnet_config_list
        self.firstConvKernelSize = ResnetCigtConstants.first_conv_kernel_size
        self.firstConvOutputDim = ResnetCigtConstants.first_conv_output_dim
        self.firstConvStride = ResnetCigtConstants.first_conv_stride
        self.bnMomentum = ResnetCigtConstants.bn_momentum
        self.batchNormType = ResnetCigtConstants.batch_norm_type
        self.applyMaskToBatchNorm = ResnetCigtConstants.apply_mask_to_batch_norm
        path_counts = [d_["path_count"] for d_ in self.resnetConfigList][1:]
        self.blockParametersDict = self.interpret_config_list()
        # self.bnMomentum = ResnetCigtConstants.bn_momentum
        self.decisionAveragePoolingStrides = ResnetCigtConstants.decision_average_pooling_strides
        self.decisionDimensions = ResnetCigtConstants.decision_dimensions
        self.startMovingAveragesFromZero = ResnetCigtConstants.start_moving_averages_from_zero
        self.informationGainBalanceCoeff = ResnetCigtConstants.information_gain_balance_coeff
        # self.informationGainBalanceCoeff = ResnetCigtConstants.information_gain_balance_coeff
        # self.optimizer = None
        # self.batchNormType = ResnetCigtConstants.batch_norm_type
        # self.applyMaskToBatchNorm = ResnetCigtConstants.apply_mask_to_batch_norm
        # self.startMovingAveragesFromZero = ResnetCigtConstants.start_moving_averages_from_zero

        super().__init__(run_id,
                         ResnetCigtConstants.batch_size,
                         ResnetCigtConstants.input_dims,
                         ResnetCigtConstants.class_count,
                         path_counts,
                         ResnetCigtConstants.softmax_decay_controller,
                         ResnetCigtConstants.learning_rate_calculator,
                         ResnetCigtConstants.optimizer_type,
                         ResnetCigtConstants.decision_non_linearity,
                         ResnetCigtConstants.decision_loss_coeff,
                         ResnetCigtConstants.routing_strategy_name,
                         ResnetCigtConstants.use_straight_through,
                         ResnetCigtConstants.warm_up_period,
                         ResnetCigtConstants.decision_drop_probability,
                         ResnetCigtConstants.classification_drop_probability,
                         ResnetCigtConstants.decision_wd,
                         ResnetCigtConstants.classification_wd,
                         ResnetCigtConstants.evaluation_period,
                         ResnetCigtConstants.measurement_start,
                         ResnetCigtConstants.save_model,
                         model_definition,
                         *args, **kwargs)

        self.build_network()
        # self.dummyBlock = None
        # self.calculate_regularization_coefficients()
        self.optimizer = self.get_optimizer()

    def build_network(self):
        self.cigtBlocks = []
        super(ResnetCigt, self).build_network()
        curr_node = self.rootNode
        assert len(self.pathCounts) == len(self.blockParametersDict)
        assert len(self.pathCounts) == len(self.decisionDimensions) + 1
        assert len(self.pathCounts) == len(self.decisionAveragePoolingStrides) + 1
        for block_id, path_count in enumerate(self.pathCounts):
            prev_block_path_count = 1 if block_id == 0 else self.pathCounts[block_id - 1]
            if block_id == 0:
                cigt_block = ResnetCigtRootBlock(
                    node=curr_node,
                    first_conv_kernel_size=self.firstConvKernelSize,
                    first_conv_output_dim=self.firstConvOutputDim,
                    first_conv_stride=self.firstConvStride,
                    block_parameters=self.blockParametersDict[block_id],
                    bn_momentum=self.bnMomentum,
                    batch_norm_type=self.batchNormType,
                    apply_mask_to_batch_norm=self.applyMaskToBatchNorm,
                    start_moving_averages_from_zero=self.startMovingAveragesFromZero,
                    class_count=self.classCount,
                    routing_strategy_name=self.routingStrategyName,
                    decision_drop_probability=self.decisionDropProbability,
                    decision_average_pooling_stride=self.decisionAveragePoolingStrides[block_id],
                    decision_dim=self.decisionDimensions[block_id],
                    decision_non_linearity=self.decisionNonLinearity,
                    ig_balance_coefficient=self.informationGainBalanceCoeff,
                    prev_block_path_count=prev_block_path_count,
                    this_block_path_count=self.pathCounts[block_id],
                    next_block_path_count=self.pathCounts[block_id + 1],
                    use_straight_through=self.useStraightThrough)
            elif 0 < block_id < len(self.pathCounts) - 1:
                cigt_block = ResnetCigtInnerBlock(
                    node=curr_node,
                    block_parameters=self.blockParametersDict[block_id],
                    bn_momentum=self.bnMomentum,
                    batch_norm_type=self.batchNormType,
                    apply_mask_to_batch_norm=self.applyMaskToBatchNorm,
                    start_moving_averages_from_zero=self.startMovingAveragesFromZero,
                    class_count=self.classCount,
                    routing_strategy_name=self.routingStrategyName,
                    decision_drop_probability=self.decisionDropProbability,
                    decision_average_pooling_stride=self.decisionAveragePoolingStrides[block_id],
                    decision_dim=self.decisionDimensions[block_id],
                    decision_non_linearity=self.decisionNonLinearity,
                    ig_balance_coefficient=self.informationGainBalanceCoeff,
                    prev_block_path_count=prev_block_path_count,
                    this_block_path_count=self.pathCounts[block_id],
                    next_block_path_count=self.pathCounts[block_id + 1],
                    use_straight_through=self.useStraightThrough)
            elif block_id == len(self.pathCounts) - 1:
                cigt_block = ResnetCigtLeafBlock(
                    node=curr_node,
                    block_parameters=self.blockParametersDict[block_id],
                    bn_momentum=self.bnMomentum,
                    batch_norm_type=self.batchNormType,
                    apply_mask_to_batch_norm=self.applyMaskToBatchNorm,
                    start_moving_averages_from_zero=self.startMovingAveragesFromZero,
                    class_count=self.classCount,
                    prev_block_path_count=prev_block_path_count,
                    this_block_path_count=self.pathCounts[block_id])
            else:
                ValueError("Unexpected block_id:{0}".format(block_id))
                return
            self.cigtBlocks.append(cigt_block)
            if curr_node.isLeaf:
                curr_node = self.dagObject.children(node=curr_node)
                assert len(curr_node) == 0
            else:
                curr_node = self.dagObject.children(node=curr_node)
                assert len(curr_node) == 1
                curr_node = curr_node[0]

    def calculate_total_macs(self):
        leaf_node = self.cigtNodes[-1]
        loss_layer_cost = leaf_node.opMacCostsDict[self.cigtBlocks[-1].lossLayer.opName]
        routed_loss_layer_cost = loss_layer_cost / self.pathCounts[-1]
        leaf_node.macCost -= loss_layer_cost
        leaf_node.macCost += routed_loss_layer_cost
        leaf_node.opMacCostsDict[self.cigtBlocks[-1].lossLayer.opName] = routed_loss_layer_cost
        super(ResnetCigt, self).calculate_total_macs()




        # for block_id, path_count in enumerate(self.pathCounts):
        #     prev_block_path_count = 1 if block_id == 0 else self.pathCounts[block_id - 1]
        #     if block_id < len(self.pathCounts) - 1:
        #         block = LeNetCigtInnerBlock(node=curr_node,
        #                                     kernel_size=self.kernelSizes[block_id],
        #                                     num_of_filters=self.filterCounts[block_id],
        #                                     strides=(1, 1),
        #                                     activation="relu",
        #                                     use_bias=True,
        #                                     padding="same",
        #                                     decision_drop_probability=self.decisionDropProbability,
        #                                     decision_dim=self.decisionDimensions[block_id],
        #                                     bn_momentum=self.bnMomentum,
        #                                     prev_block_path_count=prev_block_path_count,
        #                                     this_block_path_count=self.pathCounts[block_id],
        #                                     next_block_path_count=self.pathCounts[block_id + 1],
        #                                     ig_balance_coefficient=self.informationGainBalanceCoeff,
        #                                     class_count=self.classCount,
        #                                     routing_strategy=self.routingStrategyName,
        #                                     use_straight_through=self.useStraightThrough,
        #                                     decision_non_linearity=self.decisionNonLinearity)
        #     else:
        #         block = LeNetCigtLeafBlock(node=curr_node,
        #                                    kernel_size=self.kernelSizes[block_id],
        #                                    num_of_filters=self.filterCounts[block_id],
        #                                    activation="relu",
        #                                    hidden_layer_dims=self.hiddenLayers,
        #                                    classification_dropout_prob=self.classificationDropProbability,
        #                                    use_bias=True,
        #                                    padding="same",
        #                                    strides=(1, 1),
        #                                    class_count=self.classCount,
        #                                    prev_block_path_count=prev_block_path_count,
        #                                    this_block_path_count=self.pathCounts[block_id])
        #     self.cigtBlocks.append(block)
        #     if curr_node.isLeaf:
        #         curr_node = self.dagObject.children(node=curr_node)
        #         assert len(curr_node) == 0
        #     else:
        #         curr_node = self.dagObject.children(node=curr_node)
        #         assert len(curr_node) == 1
        #         curr_node = curr_node[0]
        # # self.build(input_shape=[(self.batchSize, *self.inputDims), (self.batchSize, )])
        #

    def interpret_config_list(self):
        block_list = []
        # Unravel the configuration information into a complete block by block list.
        for block_id, block_config_dict in enumerate(self.resnetConfigList):
            path_count = block_config_dict["path_count"]
            for idx, d_ in enumerate(block_config_dict["layer_structure"]):
                for idy in range(d_["layer_count"]):
                    block_list.append((block_id, path_count, d_["feature_map_count"]))

        block_parameters_dict = {}
        for layer_id, layer_info in enumerate(block_list):
            block_id = layer_info[0]
            path_count = layer_info[1]
            feature_map_count = layer_info[2]
            if block_id not in block_parameters_dict:
                block_parameters_dict[block_id] = []
            block_options = {}
            if layer_id == 0:
                block_options["in_dimension"] = self.firstConvOutputDim
                block_options["input_path_count"] = 1
            else:
                path_count_prev = block_list[layer_id - 1][1]
                feature_map_count_prev = block_list[layer_id - 1][2]
                block_options["in_dimension"] = feature_map_count_prev
                block_options["input_path_count"] = path_count_prev
            block_options["layer_id"] = layer_id
            block_options["block_id"] = block_id
            block_options["out_dimension"] = feature_map_count
            block_options["output_path_count"] = path_count
            if layer_id in ResnetCigtConstants.double_stride_layers:
                block_options["stride"] = 2
            else:
                block_options["stride"] = 1
            block_parameters_dict[block_id].append(block_options)
        return block_parameters_dict

    @staticmethod
    def create_default_config_json():
        root_list = []
        feature_map_counts = [16, 32, 64]
        for block_id, fm_count in enumerate(feature_map_counts):
            for idy in range(18):
                if block_id > 0 and idy == 0:
                    in_dimension = feature_map_counts[block_id - 1]
                    stride = 2
                else:
                    in_dimension = feature_map_counts[block_id]
                    stride = 1

                input_path_count = -1
                out_dimension = feature_map_counts[block_id]
                output_path_count = -1
                layer_id = len(root_list)
                root_list.append(
                    {
                        "in_dimension": in_dimension,
                        "input_path_count": input_path_count,
                        "stride": stride,
                        "out_dimension": out_dimension,
                        "output_path_count": output_path_count,
                        "layer_id": layer_id,
                        "block_id": block_id
                    }
                )

        # Serializing json
        json_object = json.dumps(root_list, indent=4)

        # Writing to sample.json
        with open("resnet_cigt_layer_config.json", "w") as outfile:
            outfile.write(json_object)
