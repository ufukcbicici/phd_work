from auxillary.db_logger import DbLogger
from tf_2_cign.cigt.cigt import Cigt
import json

from tf_2_cign.cigt.custom_layers.resnet_layers.resnet_cigt_inner_block import ResnetCigtInnerBlock
from tf_2_cign.cigt.custom_layers.resnet_layers.resnet_cigt_leaf_block import ResnetCigtLeafBlock
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
        self.doubleStrideLayers = ResnetCigtConstants.double_stride_layers
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

        if len(self.pathCounts) > 1:
            assert len(self.pathCounts) == len(self.blockParametersDict)
            assert len(self.pathCounts) == len(self.decisionDimensions) + 1
            assert len(self.pathCounts) == len(self.decisionAveragePoolingStrides) + 1
        elif len(self.pathCounts) == 1:
            assert self.pathCounts[0] == 1
            assert len(self.decisionDimensions) == 1 and self.decisionDimensions[0] == -1
            assert len(self.decisionAveragePoolingStrides) == 1 and self.decisionAveragePoolingStrides[0] == -1
        else:
            raise ValueError("0 length self.pathCounts is not valid.")

        for block_id, path_count in enumerate(self.pathCounts):
            prev_block_path_count = 1 if block_id == 0 else self.pathCounts[block_id - 1]
            # Is this the root node? If so, use the initial conv layer as the transformation.
            if block_id == 0:
                first_conv_kernel_size = self.firstConvKernelSize
                first_conv_output_dim = self.firstConvOutputDim
                first_conv_stride = self.firstConvStride
            else:
                first_conv_kernel_size = -1
                first_conv_output_dim = -1
                first_conv_stride = -1

            # Inner block
            if block_id < len(self.pathCounts) - 1:
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
                    use_straight_through=self.useStraightThrough,
                    first_conv_kernel_size=first_conv_kernel_size,
                    first_conv_output_dim=first_conv_output_dim,
                    first_conv_stride=first_conv_stride)
            else:
                cigt_block = ResnetCigtLeafBlock(
                    node=curr_node,
                    block_parameters=self.blockParametersDict[block_id],
                    bn_momentum=self.bnMomentum,
                    batch_norm_type=self.batchNormType,
                    apply_mask_to_batch_norm=self.applyMaskToBatchNorm,
                    start_moving_averages_from_zero=self.startMovingAveragesFromZero,
                    class_count=self.classCount,
                    prev_block_path_count=prev_block_path_count,
                    this_block_path_count=self.pathCounts[block_id],
                    first_conv_kernel_size=first_conv_kernel_size,
                    first_conv_output_dim=first_conv_output_dim,
                    first_conv_stride=first_conv_stride)

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
            if layer_id in self.doubleStrideLayers:
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

    def get_explanation_string(self):
        kv_rows = []
        explanation = ""
        explanation += super().get_explanation_string()
        explanation = self.add_explanation(name_of_param="firstConvKernelSize", value=self.firstConvKernelSize,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="firstConvStride", value=self.firstConvStride,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="firstConvOutputDim", value=self.firstConvOutputDim,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="decisionAveragePoolingStrides",
                                           value=self.decisionAveragePoolingStrides,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="decisionDimensions", value=self.decisionDimensions,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="applyMaskToBatchNorm", value=self.applyMaskToBatchNorm,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="startMovingAveragesFromZero",
                                           value=self.startMovingAveragesFromZero,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="batchNormType", value=self.batchNormType,
                                           explanation=explanation, kv_rows=kv_rows)
        explanation = self.add_explanation(name_of_param="doubleStrideLayers", value=self.doubleStrideLayers,
                                           explanation=explanation, kv_rows=kv_rows)
        # Explanation for block configurations
        block_params = [(block_id, block_config_list)
                        for block_id, block_config_list in self.blockParametersDict.items()]
        block_params = sorted(block_params, key=lambda tpl: tpl[0])

        layer_id = 0
        for t_ in block_params:
            block_id = t_[0]
            block_config_list = t_[1]
            for block_config_dict in block_config_list:
                explanation = self.add_explanation(name_of_param="BasicBlock_{0} in_dimension".format(layer_id),
                                                   value=block_config_dict["in_dimension"],
                                                   explanation=explanation, kv_rows=kv_rows)
                explanation = self.add_explanation(name_of_param="BasicBlock_{0} input_path_count".format(layer_id),
                                                   value=block_config_dict["input_path_count"],
                                                   explanation=explanation, kv_rows=kv_rows)
                explanation = self.add_explanation(name_of_param="BasicBlock_{0} layer_id".format(layer_id),
                                                   value=layer_id,
                                                   explanation=explanation, kv_rows=kv_rows)
                assert block_id == block_config_dict["block_id"]
                explanation = self.add_explanation(name_of_param="BasicBlock_{0} block_id".format(layer_id),
                                                   value=block_config_dict["block_id"],
                                                   explanation=explanation, kv_rows=kv_rows)
                explanation = self.add_explanation(name_of_param="BasicBlock_{0} out_dimension".format(layer_id),
                                                   value=block_config_dict["out_dimension"],
                                                   explanation=explanation, kv_rows=kv_rows)
                explanation = self.add_explanation(name_of_param="BasicBlock_{0} output_path_count".format(layer_id),
                                                   value=block_config_dict["output_path_count"],
                                                   explanation=explanation, kv_rows=kv_rows)
                explanation = self.add_explanation(name_of_param="BasicBlock_{0} stride".format(layer_id),
                                                   value=block_config_dict["stride"],
                                                   explanation=explanation, kv_rows=kv_rows)

                layer_id += 1

        DbLogger.write_into_table(rows=kv_rows, table="run_parameters")
        return explanation
