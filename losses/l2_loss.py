import tensorflow as tf
from auxillary.constants import ChannelTypes, TrainingHyperParameters
from framework.network_channel import NetworkChannel
from losses.generic_loss import GenericLoss


class L2Loss(GenericLoss):
    Name = "L2Loss"

    def __init__(self, parent_node, argument, training_program):
        super().__init__(parent_node=parent_node)
        self.argument = argument
        self.trainingProgram = training_program
        self.name = "{0}_wd".format(self.argument.name)

    def build_training_network(self):
        with NetworkChannel(parent_node=self.parentNode, parent_node_channel=ChannelTypes.loss) as loss_channel:
            wd_tensor = self.parentNode.parentNetwork.add_networkwise_input(name=self.name, channel=loss_channel,
                                                                            tensor_type=tf.float32)
            l2_loss = loss_channel.add_operation(op=tf.nn.l2_loss(self.argument.tensor))
            loss_channel.add_operation(op=(wd_tensor * l2_loss))

    def build_evaluation_network(self):
        return

    def finalize(self):
        self.isFinalized = True
