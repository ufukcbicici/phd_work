import tensorflow as tf

# Tensorflow specific code (This should be isolated at some point in future)
from auxillary.constants import OperationTypes
from framework.node_input_outputs import NetworkOutput


class NetworkChannel:
    def __init__(self, node, channel, producer_node=None, producer_channel=None, producer_channel_index=None):
        self.parentNode = node
        self.channel = channel
        if channel not in self.parentNode.networkChannels:
            self.parentNode.networkChannels[channel] = []
        self.channelIndex = len(self.parentNode.networkChannels[channel])
        if producer_node is None:
            self.producerNode = self.parentNode
            self.producerChannel = self.channel
            self.producerChannelIndex = self.channelIndex
        else:
            self.producerNode = producer_node
            self.producerChannel = producer_channel
            self.producerChannelIndex = producer_channel_index
        self.parentNode.networkChannels[channel].append(self)
        self.producerTriple = (self.producerNode, self.producerChannel, self.producerChannelIndex)

    def __enter__(self):
        return self

    # Tensorflow specific code (This should be isolated at some point in future)
    def __exit__(self, type, value, traceback):
        if type is not None:
            print("Error:{0}".format(value))
            raise Exception(value)
        output_tensor = tf.get_collection(key=self.channel.value, scope=self.parentNode.indicatorText)[-1]
        self.parentNode.outputs[self.producerTriple] = NetworkOutput(node=self.parentNode, channel=self.channel,
                                                                     channel_index=self.channelIndex,
                                                                     producer_node=self.producerNode,
                                                                     producer_channel=self.producerChannel,
                                                                     producer_channel_index=self.producerChannelIndex,
                                                                     output_object=output_tensor)

    def add_operation(self, op):
        tf.add_to_collection(self.channel.value, op)
        return op
