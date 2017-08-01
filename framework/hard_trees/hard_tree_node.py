import tensorflow as tf
from auxillary.constants import ProblemType, ChannelTypes, ArgumentTypes, ActivationType, GlobalInputNames
from auxillary.tf_layer_factory import TfLayerFactory
from framework.network_channel import NetworkChannel
from framework.network_node import NetworkNode
from losses.cross_entropy_loss import CrossEntropyLoss


class HardTreeNode(NetworkNode):
    def __init__(self, index, containing_network, is_root, is_leaf, is_accumulation):
        super().__init__(index, containing_network, is_root, is_leaf, is_accumulation=is_accumulation)

    # Tensorflow specific code (This should be isolated at some point in future)
    # This method is OK
    def attach_loss_eval_channels(self):
        if self.isLeaf and not self.isAccumulation:
            self.attach_leaf_node_loss_eval_channels()
        elif not self.isLeaf and self.isAccumulation:
            self.attach_acc_node_loss_eval_channels()
        else:
            raise Exception(
                "attach_loss_eval_channels has been called on an invalid node. "
                "self.isLeaf:{0} and self.isAccumulation:{1}".format(
                    self.isLeaf, self.isAccumulation))

    # This method is OK
    def attach_shrinkage_losses(self):
        super().attach_shrinkage_losses()

    # This method is OK
    def attach_decision(self):
        # Step 1): Gather all inputs which will enter to decision step.
        tensor_list = []
        # Get all ancestor activations, as allowed by the related hyperparameter.
        tensor_list.extend(self.get_activation_inputs())
        # Get all h operators
        tensor_list.extend(self.get_outputs_of_given_type(channel_set={ChannelTypes.h_operator}))
        # Create activation output
        with NetworkChannel(parent_node=self,
                            parent_node_channel=ChannelTypes.branching_activation) as branching_channel:
            # Concatenate all tensors
            concatenated_features = branching_channel.add_operation(op=tf.concat(tensor_list, axis=1))
            feature_dimension = concatenated_features.shape[1].value
            # Apply W'[h_operators,parent_activations] + b, where W' is the transpose of the hyperplanes and
            # b are the biases.
            activation_tensor = TfLayerFactory.create_fc_layer(node=self, channel=branching_channel,
                                                               input_tensor=concatenated_features,
                                                               fc_shape=[feature_dimension,
                                                                         self.parentNetwork.treeDegree],
                                                               init_type=self.parentNetwork.activationInit,
                                                               activation_type=ActivationType.no_activation,
                                                               post_fix="Activation")
        # Create branching probabilities
        with NetworkChannel(parent_node=self,
                            parent_node_channel=ChannelTypes.branching_probabilities) as branch_prob_channel:
            branch_probabilities_tensor = branch_prob_channel.add_operation(op=tf.nn.softmax(logits=activation_tensor))
        # Create the Nxk matrix, where N is the minibatch size and k is the branch count, which contains in its (i,j)
        # entry a boolean, indicating whether the sample i will go into the child j of this node.
        with NetworkChannel(parent_node=self,
                            parent_node_channel=ChannelTypes.branching_masks_unified) as branch_masks_unif_channel:
            branch_probability_threshold = self.parentNetwork.get_networkwise_input(
                name=GlobalInputNames.branching_prob_threshold)
            unified_branch_mask_tensor = branch_masks_unif_channel.add_operation(
                op=tf.greater_equal(x=branch_probabilities_tensor, y=branch_probability_threshold))
        # Create binary masks for each branch, which will be boolean vectors of the size (N,). k separate such vectors
        # will be generated.
        for k in self.parentNetwork.treeDegree:
            with NetworkChannel(parent_node=self,
                                parent_node_channel=ChannelTypes.branching_masks_sliced) as branch_masks_sliced_channel:
                pre_mask_tensor = branch_masks_sliced_channel.add_operation(
                    op=tf.slice(unified_branch_mask_tensor, [0, k], [-1, 1]))
                branch_masks_sliced_channel.add_operation(op=tf.reshape(pre_mask_tensor, [-1]))

    # This method is OK
    def apply_decision(self, tensor):
        parents = self.parentNetwork.dag.parents(node=self)
        if len(parents) != 1:
            raise Exception("Number of parents is not 1 in tree network.")
        parent = parents[0]
        child_index_wrt_parent = (self.index - 1) % self.parentNetwork.treeDegree
        mask_output = parent.get_output(
            producer_triple=(parent, ChannelTypes.branching_masks_sliced, child_index_wrt_parent))
        mask_tensor = mask_output.tensor
        with NetworkChannel(parent_node=self, parent_node_channel=ChannelTypes.decision) as decision_channel:
            branched_tensor = decision_channel.add_operation(op=tf.boolean_mask(tensor=tensor, mask=mask_tensor))
        return branched_tensor

    # **********************Private methods - OK**********************
    # Return all outputs of the given channel type, as stated in the set "channel_set"
    # This method is OK
    def get_outputs_of_given_type(self, channel_set):
        tensor_list = []
        for output in self.outputs.values():
            if output.producerChannel not in channel_set:
                continue
            tensor_list.append(output.tensor)
        return tensor_list

    # Get all activation inputs to this node.
    # This method is OK
    def get_activation_inputs(self):
        tensor_list = []
        # Get all ancestor activations, as allowed by the related hyperparameter.
        if self.parentNetwork.ancestorCount != 0:
            ancestors = self.parentNetwork.dag.ancestors(node=self)
            for ancestor in ancestors:
                distance = self.parentNetwork.dag.get_shortest_path_length(source=ancestor, dest=self)
                if distance <= self.parentNetwork.ancestorCount:
                    activation_tensor = \
                        self.parentNetwork.add_nodewise_input(producer_node=ancestor,
                                                              producer_channel=ChannelTypes.branching_activation,
                                                              producer_channel_index=0, dest_node=self)
                    tensor_list.append(activation_tensor)
        return tensor_list

    # This method is OK
    def attach_leaf_node_loss_eval_channels(self):
        if self.parentNetwork.problemType == ProblemType.classification:
            # Get f, h and ancestor channels, concatenate the outputs
            # Shapes are constrained to be 2 dimensional. Else, it will raise exception. We have to flatten all tensors
            # before the loss operation.
            tensor_list = []
            # First f and h channels.
            relevant_channels = {ChannelTypes.f_operator, ChannelTypes.h_operator}
            tensor_list.extend(self.get_outputs_of_given_type(channel_set=relevant_channels))
            # Then ancestor activations
            tensor_list.extend(self.get_activation_inputs())
            # Check if all tensors of the collect dimension.
            for tensor in tensor_list:
                if len(tensor.shape) != 2:
                    raise Exception("Tensors entering the loss must be 2D.")
                if tensor.shape[1].value is None:
                    raise Exception("Output tensor's dim1 cannot be None.")
            # Get the label tensor
            root_node = self.parentNetwork.nodes[0]
            if self == root_node:
                label_tensor = self.get_input(producer_triple=(None, ChannelTypes.label_input, 0))
            else:
                label_tensor = self.parentNetwork.add_nodewise_input(producer_node=root_node,
                                                                     producer_channel=ChannelTypes.label_input,
                                                                     producer_channel_index=0, dest_node=self)
            class_count = self.parentNetwork.dataset.get_label_count()
            cross_entropy_loss = CrossEntropyLoss(parent_node=self, feature_list=tensor_list, label_tensor=label_tensor,
                                                  class_count=class_count)
            NetworkNode.apply_loss(loss=cross_entropy_loss)
        else:
            raise NotImplementedError()

    # This method is OK
    def attach_acc_node_loss_eval_channels(self):
        # Step 1) Build the final loss layer.
        # Accumulate all losses from all nodes.
        loss_tensors = []
        for node in self.parentNetwork.nodes.values():
            for loss_object in node.losses.values():
                if loss_object.lossOutputs is None:
                    continue
                loss_object.lossIndex = len(loss_tensors)
                loss_tensors.extend(loss_object.lossOutputs)
        # Step 2) Add all loss tensors
        with NetworkChannel(parent_node=self, parent_node_channel=ChannelTypes.total_loss) as total_loss_channel:
            total_loss = total_loss_channel.add_operation(op=tf.add_n(loss_tensors))
        # Step 3) Gather all learnable parameters
        learnable_parameters = []
        for node in self.parentNetwork.nodes.values():
            if node == self:
                continue
            for argument in node.argumentsDict.values():
                if argument.argumentType == ArgumentTypes.learnable_parameter:
                    argument.gradientIndex = len(learnable_parameters)
                    learnable_parameters.append(argument.tensor)
        # Step 4) Calculate the gradients with respect to parameter
        with NetworkChannel(parent_node=self, parent_node_channel=ChannelTypes.gradient) as gradient_channel:
            gradients = gradient_channel.add_operation(op=tf.gradients(total_loss, learnable_parameters))
        # Step 5) Gather all tensors to be evaluated
        eval_tensors = []
        for node in self.parentNetwork.nodes.values():
            for loss_object in node.losses.values():
                if loss_object.evalOutputs is None:
                    continue
                loss_object.evalIndex = len(eval_tensors)
                eval_tensors.extend(loss_object.evalOutputs)
        # Step 6) Save the gathered loss and eval tensors, so they can re-used by the network.
        # Training
        self.parentNetwork.trainingTensorsList = []
        self.parentNetwork.trainingTensorsList.extend(loss_tensors)
        self.parentNetwork.trainingTensorsList.append(total_loss)
        self.parentNetwork.trainingTensorsList.append(gradients)
        # Evaluation
        self.parentNetwork.evaluationTensorsList = []
        self.parentNetwork.evaluationTensorsList.extend(eval_tensors)
    # **********************Private methods - OK**********************