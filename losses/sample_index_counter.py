from auxillary.constants import ChannelTypes
from framework.network_channel import NetworkChannel
from losses.generic_loss import GenericLoss


class SampleIndexCounter(GenericLoss):
    Name = "SampleIndexCounter"

    def __init__(self, parent_node):
        super().__init__(parent_node=parent_node, name=SampleIndexCounter.Name)

    def build_training_network(self):
        pass

    def build_evaluation_network(self):
        if self.parentNode.isRoot:
            # First add as a normal input to the root node, then add into the evaluation channel.
            sample_index_tensor = self.parentNode.parentNetwork.add_nodewise_input(
                producer_channel=ChannelTypes.indices_input, dest_node=self.parentNode)
        else:
            sample_index_tensor = self.parentNode.parentNetwork.add_nodewise_input(
                producer_node=self.parentNode.parentNetwork.get_root_node(),
                producer_channel=ChannelTypes.indices_input, dest_node=self.parentNode)
        with NetworkChannel(parent_node=self.parentNode,
                            parent_node_channel=ChannelTypes.evaluation) as eval_channel:
            eval_channel.add_operation(op=sample_index_tensor)
            self.evalOutput = sample_index_tensor

    def finalize(self):
        self.isFinalized = True
