import tensorflow as tf
from auxillary.constants import ChannelTypes, LossType
from framework.network_channel import NetworkChannel
from losses.generic_loss import GenericLoss


class SampleIndexCounter(GenericLoss):
    Name = "SampleIndexCounter"

    def __init__(self, parent_node):
        super().__init__(parent_node=parent_node, name=SampleIndexCounter.Name, loss_type=LossType.eval_term,
                         is_differentiable=False)
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
            self.lossOutputs = [sample_count_tensor]

    def build_evaluation_network(self):
        with NetworkChannel(parent_node=self.parentNode,
                            parent_node_channel=ChannelTypes.evaluation) as eval_channel:
            eval_channel.add_operation(op=self.sampleIndexTensor)
            self.evalOutputs = [self.sampleIndexTensor]

    def finalize(self):
        super().finalize()
