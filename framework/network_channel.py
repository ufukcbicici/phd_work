import tensorflow as tf

# Tensorflow specific code (This should be isolated at some point in future)
from auxillary.constants import OperationTypes


class NetworkChannel:
    def __init__(self, node, channel_name):
        self.parentNode = node
        self.channelName = channel_name
        if channel_name not in self.parentNode.networkChannels:
            self.parentNode.networkChannels[channel_name] = []
        self.channelIndex = len(self.parentNode.networkChannels[channel_name])
        self.parentNode.networkChannels[channel_name].append(self)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def add_operation(self, op):
        tf.add_to_collection(self.channelName, op)
        return op
