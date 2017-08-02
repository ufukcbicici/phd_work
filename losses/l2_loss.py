import tensorflow as tf
from auxillary.constants import ChannelTypes, TrainingHyperParameters
from framework.network_channel import NetworkChannel
from losses.generic_loss import GenericLoss


class L2Loss(GenericLoss):
    Name = "L2Loss"

    def __init__(self, parent_node, argument, training_program):
        self.argument = argument
        self.trainingProgram = training_program
        super().__init__(parent_node=parent_node, name="{0}_wd".format(self.argument.name))

    def build_training_network(self):
        wd_tensor = self.parentNode.parentNetwork.add_networkwise_input(name=self.name, tensor_type=tf.float32)
        with NetworkChannel(parent_node=self.parentNode, parent_node_channel=ChannelTypes.loss) as loss_channel:
            l2_loss = loss_channel.add_operation(op=tf.nn.l2_loss(self.argument.tensor))
            self.lossOutputs = [wd_tensor * l2_loss]
            loss_channel.add_operation(op=(self.lossOutputs[0]))

    def build_evaluation_network(self):
        return

    def finalize(self):
        super().finalize()
