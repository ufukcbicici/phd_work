import tensorflow as tf
from auxillary.constants import ChannelTypes
from auxillary.tf_layer_factory import TfLayerFactory
from framework.network_channel import NetworkChannel
from losses.generic_loss import GenericLoss


class CrossEntropyLoss(GenericLoss):
    Name = "CrossEntropyLoss"
    Features = 0
    LabelTensor = 1
    ClassCount = 2

    def __init__(self, parent_node, feature_list, label_tensor, class_count):
        super().__init__(parent_node=parent_node)
        self.logitTensor = None
        self.featureList = feature_list
        self.labelTensor = label_tensor
        self.classCount = class_count
        # Pre-Loss channel
        with NetworkChannel(node=self.parentNode, channel=ChannelTypes.pre_loss) as pre_loss_channel:
            if len(self.featureList) > 1:
                final_feature = pre_loss_channel.add_operation(op=tf.concat(values=self.featureList, axis=1))
            elif len(self.featureList) == 1:
                final_feature = self.featureList[0]
            else:
                raise Exception("No features have been passed to cross entropy loss.")
            final_dimension = final_feature.shape[1].value
            self.logitTensor = TfLayerFactory.create_fc_layer(node=self.parentNode, channel=pre_loss_channel,
                                                              input_tensor=final_feature,
                                                              fc_shape=[final_dimension, self.classCount],
                                                              init_type=self.parentNode.parentNetwork.lossLayerInit,
                                                              activation_type=self.parentNode.parentNetwork.
                                                              lossLayerActivation,
                                                              post_fix="pre_loss")

    def build_training_network(self):
        if self.logitTensor is None:
            raise Exception("No logit tensor have been found.")
        with NetworkChannel(node=self.parentNode, channel=ChannelTypes.loss,
                            channel_name=CrossEntropyLoss.Name) as loss_channel:
            softmax_cross_entropy = loss_channel.add_operation(
                op=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labelTensor, logits=self.logitTensor))
            self.lossOutput = loss_channel.add_operation(op=tf.reduce_mean(input_tensor=softmax_cross_entropy))

    def build_evaluation_network(self):
        if self.logitTensor is None:
            raise Exception("No logit tensor have been found.")
        with NetworkChannel(node=self.parentNode, channel=ChannelTypes.evaluation) as eval_channel:
            posterior_probs = eval_channel.add_operation(op=tf.nn.softmax(logits=self.logitTensor))
            argmax_label_prediction = eval_channel.add_operation(op=tf.argmax(posterior_probs, 1))
            comparison_with_labels = eval_channel.add_operation(
                op=tf.equal(x=argmax_label_prediction, y=self.labelTensor))
            comparison_cast = eval_channel.add_operation(op=tf.cast(comparison_with_labels, tf.float32))
            self.evalOutput = eval_channel.add_operation(op=tf.reduce_mean(input_tensor=comparison_cast))
            # TODO: Return only the number of correct samples and return the number of all samples in this node as well

    def finalize(self):
        super().finalize()
        self.featureList = None
        self.labelTensor = None
