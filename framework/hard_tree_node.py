import tensorflow as tf
from auxillary.constants import ProblemType, ChannelTypes
from auxillary.tf_layer_factory import TfLayerFactory
from framework.network_channel import NetworkChannel
from framework.network_node import NetworkNode
from losses.cross_entropy_loss import CrossEntropyLoss


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
            relevant_channels = {ChannelTypes.f_operator, ChannelTypes.h_operator,
                                 ChannelTypes.ancestor_activation}
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
                label_tensor = self.get_input(producer_node=None, channel=ChannelTypes.label_input, channel_index=0)
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
