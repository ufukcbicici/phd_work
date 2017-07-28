from auxillary.constants import ChannelTypes
from losses.generic_loss import GenericLoss


class SampleIndexCounter(GenericLoss):
    Name = "SampleIndexCounter"

    def __init__(self, parent_node):
        super().__init__(parent_node=parent_node)

    def build_training_network(self):
        pass

    def build_evaluation_network(self):
        if self.parentNode.isRoot:
            self.parentNode.parentNetwork.add_nodewise_input(
                producer_channel=ChannelTypes.indices_input, dest_node=self.parentNode)
        else:
            self.parentNode.parentNetwork.add_nodewise_input(
                producer_node=self.parentNode.parentNetwork.get_root_node(),
                producer_channel=ChannelTypes.indices_input, dest_node=self.parentNode)

    def finalize(self):
        self.isFinalized = True
