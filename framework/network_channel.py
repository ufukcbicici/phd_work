import tensorflow as tf

# Tensorflow specific code (This should be isolated at some point in future)
from auxillary.constants import ChannelTypes, GlobalInputNames
from auxillary.runtime import Runtime
from framework.node_input_outputs import NetworkIOObject


class NetworkChannel:
    def __init__(self, parent_node, parent_node_channel, producer_node=None, producer_channel=None,
                 producer_channel_index=None, channel_name=None):
        self.parentNode = parent_node
        self.channel = parent_node_channel
        if parent_node_channel not in self.parentNode.networkChannels:
            self.parentNode.networkChannels[parent_node_channel] = []
        self.channelIndex = len(self.parentNode.networkChannels[parent_node_channel])
        if producer_node is None:
            self.producerNode = self.parentNode
            self.producerChannel = self.channel
            self.producerChannelIndex = self.channelIndex
        else:
            self.producerNode = producer_node
            self.producerChannel = producer_channel
            self.producerChannelIndex = producer_channel_index
        self.parentNode.networkChannels[parent_node_channel].append(self)
        self.channelName = "{0}_{1}".format(self.channel.value, self.channelIndex)
        self.producerTriple = (self.producerNode, self.producerChannel, self.producerChannelIndex)

    def __enter__(self):
        Runtime.push_node(node=self.parentNode)
        return self

    # Tensorflow specific code (This should be isolated at some point in future)
    def __exit__(self, type, value, traceback):
        if type is not None:
            print("Error:{0}".format(value))
            raise Exception(value)
        if self.producerTriple in self.parentNode.outputs:
            raise Exception("The triple {0} already exists in the outputs.".format(self.producerTriple))
        context_name = Runtime.get_context_name()
        output_tensor = tf.get_collection(key=self.channelName, scope=context_name)[-1]
        output_object = NetworkIOObject(producer_node=self.producerNode,
                                        producer_channel=self.producerChannel,
                                        producer_channel_index=self.producerChannelIndex,
                                        tensor=output_tensor)
        self.parentNode.add_output(producer_triple=self.producerTriple, output_object=output_object)
        Runtime.pop_node()

    def add_operation(self, op):
        tf.add_to_collection(self.channelName, op)
        return op
