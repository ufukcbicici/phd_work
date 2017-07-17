class NetworkInput:
    def __init__(self, source_node, source_channel, source_channel_index,
                 producer_node, producer_channel, producer_channel_index,
                 input_object):
        self.inputObject = input_object
        self.sourceNode = source_node
        self.sourceChannel = source_channel
        self.sourceChannelIndex = source_channel_index
        # This triple indicates which node actually produces that output. All other nodes in between only applies
        # decision transformations on that output.
        self.producerNode = producer_node
        self.producerChannel = producer_channel
        self.producerChannelIndex = producer_channel_index


class NetworkOutput:
    def __init__(self, node, channel, channel_index,
                 producer_node, producer_channel, producer_channel_index,
                 output_object):
        self.outputObject = output_object
        # This triple indicates directly from which channel and channel index this output goes through.
        self.currentNode = node
        self.currentChannel = channel
        self.currentChannelIndex = channel_index
        # This triple indicates to which node, which channel and which channel index this output belongs.
        self.producerNode = producer_node
        self.producerChannel = producer_channel
        self.producerChannelIndex = producer_channel_index

    def produce_input(self):
        input_object = NetworkInput(source_node=self.currentNode.currentNode,
                                    source_channel=self.currentNode.currentChannel,
                                    source_channel_index=self.currentNode.currentChannelIndex,
                                    input_object=self.outputObject,
                                    producer_node=self.producerNode,
                                    producer_channel=self.producerChannel,
                                    producer_channel_index=self.producerChannelIndex)
        return input_object