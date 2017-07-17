import tensorflow as tf

# Tensorflow specific code (This should be isolated at some point in future)
from auxillary.constants import OperationTypes
from framework.network_node import NetworkOutput


class NetworkChannel:
    def __init__(self, node, channel_name, producer_node=None, producer_channel=None, producer_channel_index=None):
        self.parentNode = node
        self.channelName = channel_name
        if channel_name not in self.parentNode.networkChannels:
            self.parentNode.networkChannels[channel_name] = []
        self.channelIndex = len(self.parentNode.networkChannels[channel_name])
        if producer_node is None:
            self.producerNode = self.parentNode
            self.producerChannel = self.channelName
            self.producerChannelIndex = self.channelIndex
        else:
            self.producerNode = producer_node
            self.producerChannel = producer_channel
            self.producerChannelIndex = producer_channel_index
        self.parentNode.networkChannels[channel_name].append(self)
        self.producerTriple = (self.producerNode, self.producerChannel, self.producerChannelIndex)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        output_tensor = tf.get_collection(key=self.channelName, scope=self.parentNode.self.indicatorText)
        self.parentNode.outputs[self.producerTriple] = NetworkOutput(node=self.parentNode, channel=self.channelName,
                                                                     channel_index=self.channelIndex,
                                                                     producer_node=self.producerNode,
                                                                     producer_channel=self.producerChannel,
                                                                     producer_channel_index=self.producerChannelIndex,
                                                                     output_object=output_tensor)

    def add_operation(self, op):
        tf.add_to_collection(self.channelName, op)
        return op
