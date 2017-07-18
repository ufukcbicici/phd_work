import tensorflow as tf
from auxillary.constants import ProblemType, OperationTypes
from auxillary.tf_layer_factory import TfLayerFactory
from framework.network_channel import NetworkChannel
from framework.network_node import NetworkNode


class HardTreeNode(NetworkNode):
    def __init__(self, index, containing_network, is_root, is_leaf):
        super().__init__(index, containing_network, is_root, is_leaf)

    # Tensorflow specific code (This should be isolated at some point in future)
    def attach_loss_eval_channels(self):
        if self.parentNetwork.problemType == ProblemType.classification:
            # Get f, h and ancestor channels, concatenate the outputs
            # Shapes are constrained to be 2 dimensional. Else, it will raise exception. We have to flatten all tensors
            # before the loss operation.
            tensor_list = []
            relevant_channels = {OperationTypes.f_operator, OperationTypes.h_operator,
                                 OperationTypes.ancestor_activation}
            for output in self.outputs.values():
                if output.currentChannel not in relevant_channels:
                    continue
                output_tensor = output.outputObject
                if len(output_tensor.shape) != 2:
                    raise Exception("Tensors entering the loss must be 2D.")
                if output_tensor.shape[1].value is None:
                    raise Exception("Output tensor's dim1 cannot be None.")
                tensor_list.append(output_tensor)
            # Get the label tensor
            root_node = self.parentNetwork.nodes[0]
            if self == root_node:
                label_tensor = self.get_input(producer_node=None, channel=OperationTypes.label_input, channel_index=0)
            else:
                label_tensor = self.parentNetwork.add_input(producer_node=root_node,
                                                            producer_channel=OperationTypes.label_input,
                                                            producer_channel_index=0, dest_node=self)
            class_count = self.parentNetwork.dataset.get_label_count()
            # Pre-Loss channel
            with NetworkChannel(node=self, channel=OperationTypes.pre_loss) as pre_loss_channel:
                final_feature = pre_loss_channel.add_operation(
                    op=tf.concat(values=tensor_list, axis=1))
                final_dimension = final_feature.shape[1].value
                logits = TfLayerFactory.create_fc_layer(node=self, channel=pre_loss_channel,
                                                        input_tensor=final_feature,
                                                        fc_shape=[final_dimension, class_count],
                                                        init_type=self.parentNetwork.lossLayerInit,
                                                        activation_type=self.parentNetwork.lossLayerActivation,
                                                        post_fix="pre_loss")
            # Loss channel
            with NetworkChannel(node=self, channel=OperationTypes.loss) as loss_channel:
                softmax_cross_entropy = loss_channel.add_operation(
                    op=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_tensor, logits=logits))
                loss_channel.add_operation(
                    op=tf.reduce_mean(input_tensor=softmax_cross_entropy))
            # Evaluation channel
            with NetworkChannel(node=self, channel=OperationTypes.evaluation) as eval_channel:
                posterior_probs = eval_channel.add_operation(op=tf.nn.softmax(logits=logits))
                argmax_label_prediction = eval_channel.add_operation(op=tf.argmax(posterior_probs, 1))
                comparison_with_labels = eval_channel.add_operation(
                    op=tf.equal(x=argmax_label_prediction, y=label_tensor))
                comparison_cast = eval_channel.add_operation(op=tf.cast(comparison_with_labels, tf.float32))
                eval_channel.add_operation(op=tf.reduce_mean(input_tensor=comparison_cast))
        else:
            raise NotImplementedError()
