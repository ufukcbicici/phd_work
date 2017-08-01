import tensorflow as tf

# Tensorflow specific code (This should be isolated at some point in future)
from auxillary.constants import ChannelTypes, GlobalInputNames
from framework.node_input_outputs import NetworkIOObject

# TODO: SOLVE THE ISSUE WITH GLOBAL NAMES!!!
class NetworkChannel:
    def __init__(self, parent_node, parent_node_channel, producer_node=None, producer_channel=None,
                 producer_channel_index=None, channel_name=None):
        self.parentNode = parent_node
        self.channel = parent_node_channel
        if self.parentNode is not None:
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
            if channel_name is None:
                self.channelName = "{0}_{1}".format(self.channel.value, self.channelIndex)
            else:
                self.channelName = channel_name
        else:
            self.producerNode = None
            self.producerChannel = self.channel
            self.channelIndex = -1
            self.producerChannelIndex = -1
            self.channelName = GlobalInputNames.global_scope.value
        self.producerTriple = (self.producerNode, self.producerChannel, self.producerChannelIndex)

    def __enter__(self):
        return self

    # Tensorflow specific code (This should be isolated at some point in future)
    def __exit__(self, type, value, traceback):
        if type is not None:
            print("Error:{0}".format(value))
            raise Exception(value)
        if self.parentNode is not None:
            if self.producerTriple in self.parentNode.outputs:
                raise Exception("The triple {0} already exists in the outputs.".format(self.producerTriple))
            output_tensor = tf.get_collection(key=self.channelName, scope=self.parentNode.indicatorText)[-1]
            output_object = NetworkIOObject(producer_node=self.producerNode,
                                            producer_channel=self.producerChannel,
                                            producer_channel_index=self.producerChannelIndex,
                                            tensor=output_tensor)
            self.parentNode.add_output(producer_triple=self.producerTriple, output_object=output_object)

    def add_operation(self, op):
        tf.add_to_collection(self.channelName, op)
        return op
