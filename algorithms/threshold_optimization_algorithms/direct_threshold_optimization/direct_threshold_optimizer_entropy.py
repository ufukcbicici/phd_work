import tensorflow as tf

from algorithms.threshold_optimization_algorithms.direct_threshold_optimization.direct_threshold_optimizer import \
    DirectThresholdOptimizer
from simple_tf.global_params import GlobalConstants


class DirectThresholdOptimizerEntropy(DirectThresholdOptimizer):
    def __init__(self, network, routing_data, seed):
        super().__init__(network, routing_data, seed)
        self.entropies = {}
        self.kind = "entropy"

    def threshold_test(self, node, routing_probs):
        # Hard
        self.thresholds[node.index] = tf.placeholder(dtype=tf.float32,
                                                     name="entropy_threshold{0}".format(node.index))
        # Calculate routing probability entropy
        log_prob = tf.log(routing_probs + GlobalConstants.INFO_GAIN_LOG_EPSILON)
        prob_log_prob = routing_probs * log_prob
        self.entropies[node.index] = -1.0 * tf.reduce_sum(prob_log_prob, axis=1)
        comparison_arr = tf.cast(self.entropies[node.index] >= self.thresholds[node.index], tf.bool)
        arg_max_selections = tf.one_hot(tf.argmax(routing_probs, axis=1), len(self.network.dagObject.children(node)))
        both_selections = tf.ones_like(arg_max_selections)
        thresholds = tf.where(comparison_arr, both_selections, arg_max_selections)
        return thresholds

    def get_run_dict(self):
        run_dict = {"accuracy": self.accuracy,
                    "predictedLabels": self.predictedLabels,
                    "gtLabels": self.gtLabels,
                    "finalPosteriors": self.finalPosteriors,
                    "weightsArray": self.weightsArray,
                    "weightedPosteriors": self.weightedPosteriors,
                    "posteriorsTensor": self.posteriorsTensor,
                    "selectionWeights": self.selectionWeights,
                    "pathScores": self.pathScores,
                    "thresholdTests": self.thresholdTests,
                    "routingProbabilities": self.routingProbabilities,
                    "routingProbabilitiesUncalibrated": self.routingProbabilitiesUncalibrated,
                    "branchingLogits": self.branchingLogits,
                    "thresholds": self.thresholds,
                    "powersOfTwoArr": self.powersOfTwoArr,
                    "activationCodes": self.activationCodes,
                    "selectionTuples": self.selectionTuples,
                    "networkActivationCosts": self.networkActivationCosts,
                    "activationCostsArr": self.activationCostsArr,
                    "meanActivationCost": self.meanActivationCost,
                    "score": self.score,
                    "correctnessVector": self.correctnessVector,
                    "entropies": self.entropies}
        return run_dict
