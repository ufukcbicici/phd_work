class NetworkIOObject:
    def __init__(self, producer_node, producer_channel, producer_channel_index,
                 tensor):
        self.tensor = tensor
        # This triple indicates which node actually produces that output. All other nodes in between only applies
        # decision transformations on that output.
        self.producerNode = producer_node
        self.producerChannel = producer_channel
        self.producerChannelIndex = producer_channel_index
