import tensorflow as tf
from auxillary.constants import ChannelTypes, GlobalInputNames, LossType
from auxillary.tf_layer_factory import TfLayerFactory
from framework.network_channel import NetworkChannel
from losses.generic_loss import GenericLoss


class CrossEntropyLoss(GenericLoss):
    Name = "CrossEntropyLoss"
    Features = 0
    LabelTensor = 1
    ClassCount = 2

    def __init__(self, parent_node, feature_list, label_tensor, class_count):
        super().__init__(parent_node=parent_node, loss_type=LossType.objective, is_differentiable=True)
        self.logitTensor = None
        self.featureList = feature_list
        self.labelTensor = label_tensor
        self.classCount = class_count
        # Pre-Loss channel
        with NetworkChannel(parent_node=self.parentNode, parent_node_channel=ChannelTypes.pre_loss) as pre_loss_channel:
            if len(self.featureList) > 1:
                final_feature = pre_loss_channel.add_operation(op=tf.concat(values=self.featureList, axis=1))
            elif len(self.featureList) == 1:
                final_feature = self.featureList[0]
            else:
                raise Exception("No features have been passed to cross entropy objective_loss.")
            final_dimension = final_feature.shape[1].value
            self.logitTensor = TfLayerFactory.create_fc_layer(node=self.parentNode, channel=pre_loss_channel,
                                                              input_tensor=final_feature,
                                                              fc_shape=[final_dimension, self.classCount],
                                                              init_type=self.parentNode.parentNetwork.lossLayerInit,
                                                              activation_type=self.parentNode.parentNetwork.
                                                              lossLayerActivation,
                                                              post_fix=ChannelTypes.pre_loss.value)

    def get_name(self):
        return "{0}_Node{1}".format(CrossEntropyLoss.Name, self.parentNode.index)

    @staticmethod
    def get_loss_name(node):
        return "{0}_Node{1}".format(CrossEntropyLoss.Name, node.index)

    def build_training_network(self):
        if self.logitTensor is None:
            raise Exception("No logit tensor have been found.")
        with NetworkChannel(parent_node=self.parentNode, parent_node_channel=ChannelTypes.objective_loss,
                            channel_name=CrossEntropyLoss.Name) as loss_channel:
            softmax_cross_entropy = loss_channel.add_operation(
                op=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labelTensor, logits=self.logitTensor))
            total_softmax_cross_entropy = loss_channel.add_operation(
                op=tf.reduce_sum(input_tensor=softmax_cross_entropy))
            batch_size = self.parentNode.parentNetwork.get_networkwise_input(name=GlobalInputNames.batch_size.value)
            avg_softmax_cross_entropy = loss_channel.add_operation(op=(total_softmax_cross_entropy / batch_size))
            self.lossOutputs = [avg_softmax_cross_entropy]
            self.auxOutputs = [total_softmax_cross_entropy]

    def build_evaluation_network(self):
        self.evalOutputs = []
        if self.logitTensor is None:
            raise Exception("No logit tensor have been found.")
        # Calculate the number of correct sample inferences
        with NetworkChannel(parent_node=self.parentNode,
                            parent_node_channel=ChannelTypes.evaluation) as correct_count_channel:
            posterior_probs = correct_count_channel.add_operation(op=tf.nn.softmax(logits=self.logitTensor))
            argmax_label_prediction = correct_count_channel.add_operation(op=tf.argmax(posterior_probs, 1))
            comparison_with_labels = correct_count_channel.add_operation(
                op=tf.equal(x=argmax_label_prediction, y=self.labelTensor))
            comparison_cast = correct_count_channel.add_operation(op=tf.cast(comparison_with_labels, tf.float32))
            self.evalOutputs.append(correct_count_channel.add_operation(op=tf.reduce_sum(input_tensor=comparison_cast)))
        # Calculate the total number of samples
        with NetworkChannel(parent_node=self.parentNode,
                            parent_node_channel=ChannelTypes.evaluation) as total_count_channel:
            self.evalOutputs.append(total_count_channel.add_operation(op=tf.size(input=comparison_cast)))

    def finalize(self):
        super().finalize()
        self.featureList = None
        self.labelTensor = None
