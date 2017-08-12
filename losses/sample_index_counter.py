import tensorflow as tf
from auxillary.constants import ChannelTypes, LossType
from framework.network_channel import NetworkChannel
from losses.generic_loss import GenericLoss


class SampleIndexCounter(GenericLoss):
    Name = "SampleIndexCounter"

    def __init__(self, parent_node):
        super().__init__(parent_node=parent_node, loss_type=LossType.eval_term, is_differentiable=False)
        self.sampleIndexTensor = None
        if self.parentNode.isRoot:
            # First add as a normal input to the root node, then add into the evaluation channel.
            self.sampleIndexTensor = self.parentNode.parentNetwork.add_nodewise_input(
                producer_channel=ChannelTypes.indices_input, dest_node=self.parentNode)
        else:
            self.sampleIndexTensor = self.parentNode.parentNetwork.add_nodewise_input(
                producer_node=self.parentNode.parentNetwork.get_root_node(),
                producer_channel=ChannelTypes.indices_input, dest_node=self.parentNode)
        with NetworkChannel(parent_node=self.parentNode,
                            parent_node_channel=ChannelTypes.pre_loss) as pre_loss_channel:
            pre_loss_channel.add_operation(op=self.sampleIndexTensor)

    def build_training_network(self):
        with NetworkChannel(parent_node=self.parentNode,
                            parent_node_channel=ChannelTypes.non_differentiable_loss) as non_dif_loss_channel:
            sample_count_tensor = non_dif_loss_channel.add_operation(op=tf.size(input=self.sampleIndexTensor))
            if self.parentNode != self.parentNode.parentNetwork.get_root_node():
                label_tensor = self.parentNode.parentNetwork.add_nodewise_input(
                    producer_node=self.parentNode.parentNetwork.get_root_node(),
                    producer_channel=ChannelTypes.label_input,
                    producer_channel_index=0, dest_node=self.parentNode)
                self.lossOutputs = [sample_count_tensor, self.sampleIndexTensor, label_tensor]
            else:
                self.lossOutputs = [sample_count_tensor, self.sampleIndexTensor]

    def build_evaluation_network(self):
        with NetworkChannel(parent_node=self.parentNode,
                            parent_node_channel=ChannelTypes.evaluation) as eval_channel:
            eval_channel.add_operation(op=self.sampleIndexTensor)
            if not self.parentNode.isLeaf:
                branch_probs = self.parentNode.get_output(
                    producer_triple=(self.parentNode, ChannelTypes.branching_probabilities, 0)).tensor
                self.evalOutputs = [self.sampleIndexTensor, branch_probs]
            else:
                self.evalOutputs = [self.sampleIndexTensor]

    def finalize(self):
        super().finalize()

    def get_name(self):
        return "{0}_Node{1}".format(SampleIndexCounter.Name, self.parentNode.index)

    @staticmethod
    def get_loss_name(node):
        return "{0}_Node{1}".format(SampleIndexCounter.Name, node.index)
