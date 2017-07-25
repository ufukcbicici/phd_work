import tensorflow as tf
from auxillary.constants import ProblemType, ChannelTypes, ArgumentTypes
from auxillary.tf_layer_factory import TfLayerFactory
from framework.network_channel import NetworkChannel
from framework.network_node import NetworkNode
from losses.cross_entropy_loss import CrossEntropyLoss


class HardTreeNode(NetworkNode):
    def __init__(self, index, containing_network, is_root, is_leaf, is_accumulation):
        super().__init__(index, containing_network, is_root, is_leaf, is_accumulation=is_accumulation)

    # Tensorflow specific code (This should be isolated at some point in future)
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

    def attach_shrinkage_losses(self):
        super().attach_shrinkage_losses()

    # Private methods - OK
    def attach_leaf_node_loss_eval_channels(self):
        if self.parentNetwork.problemType == ProblemType.classification:
            # Get f, h and ancestor channels, concatenate the outputs
            # Shapes are constrained to be 2 dimensional. Else, it will raise exception. We have to flatten all tensors
            # before the loss operation.
            tensor_list = []
            relevant_channels = {ChannelTypes.f_operator, ChannelTypes.h_operator,
                                 ChannelTypes.ancestor_activation}
            for output in self.outputs.values():
                if output.producerChannel not in relevant_channels:
                    continue
                output_tensor = output.tensor
                if len(output_tensor.shape) != 2:
                    raise Exception("Tensors entering the loss must be 2D.")
                if output_tensor.shape[1].value is None:
                    raise Exception("Output tensor's dim1 cannot be None.")
                tensor_list.append(output_tensor)
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

    def attach_acc_node_loss_eval_channels(self):
        # Step 1) Build the final loss layer.
        # Accumulate all losses from all nodes.
        loss_list = []
        for node in self.parentNetwork.nodes.values():
            if node == self:
                continue
            for output in node.outputs.values():
                if output.producerChannel != ChannelTypes.loss:
                    continue
                loss_list.append(output.tensor)
        # Accumulate all learnable parameters from all nodes.
        learnable_parameters = []
        for node in self.parentNetwork.nodes.values():
            if node == self:
                continue
            for argument in node.argumentsDict.values():
                if argument.argumentType == ArgumentTypes.learnable_parameter:
                    argument.gradientIndex = len(learnable_parameters)
                    learnable_parameters.append(argument.tensor)
        # Add them together and calculate the gradient of the total loss with respect to all learnable parameters.
        with NetworkChannel(parent_node=self, parent_node_channel=ChannelTypes.loss) as loss_channel:
            total_loss = loss_channel.add_operation(op=tf.add_n(loss_list))
        with NetworkChannel(parent_node=self, parent_node_channel=ChannelTypes.gradient) as gradient_channel:
            gradient_channel.add_operation(op=tf.gradients(total_loss, learnable_parameters))
            # Step 2) Build the final evaluation layer.
            # TODO: Complete this

    def attach_decision(self):
        # Step 1): Gather all inputs which will enter to decision step.
        tensor_list = []
        # Get all ancestor activations, as allowed by the related hyperparameter.
        if self.parentNetwork.ancestorCount != 0:
            ancestors = self.parentNetwork.dag.ancestors(node=self)
            for ancestor in ancestors:
                distance = self.parentNetwork.dag.get_shortest_path_length(source=ancestor, dest=self)
                if distance <= self.parentNetwork.ancestorCount:
                    activation_tensor = \
                        self.parentNetwork.add_nodewise_input(producer_node=ancestor,
                                                              producer_channel=ChannelTypes.ancestor_activation,
                                                              producer_channel_index=0, dest_node=self)
