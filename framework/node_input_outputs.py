class NetworkIOObject:
    def __init__(self, producer_node, producer_channel, producer_channel_index,
                 tensor):
        self.tensor = tensor
        # This triple indicates which node actually produces that output. All other nodes in between only applies
        # decision transformations on that output.
        self.producerNode = producer_node
        self.producerChannel = producer_channel
        self.producerChannelIndex = producer_channel_index

    def clone(self):
        copy_obj = NetworkIOObject(tensor=self.tensor,
                                   producer_node=self.producerNode,
                                   producer_channel=self.producerChannel,
                                   producer_channel_index=self.producerChannelIndex)
        return copy_obj
