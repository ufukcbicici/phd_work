import tensorflow as tf
import os

from auxillary.db_logger import DbLogger
from auxillary.parameters import DiscreteParameter
from tf_2_cign.cigt.data_classes.multipath_routing_info import MultipathCombinationInfo
from tf_2_cign.cigt.lenet_cigt import LenetCigt
from tf_2_cign.softmax_decay_algorithms.step_wise_decay_algorithm import StepWiseDecayAlgorithm
from tf_2_cign.utilities.fashion_net_constants import FashionNetConstants


class CrossEntropySearchOptimizer(object):
    def __init__(self, num_of_epochs, accuracy_mac_balance_coeff, model_loader,
                 model_id, val_ratio,
                 entropy_interval_counts, image_output_path):
        self.numOfEpochs = num_of_epochs
        self.accuracyMacBalanceCoeff = accuracy_mac_balance_coeff
        self.modelLoader = model_loader
        self.modelId = model_id
        self.valRatio = val_ratio
        self.entropyIntervalCounts = entropy_interval_counts
        self.imageOutputPath = image_output_path
        self.model, self.dataset = self.modelLoader.get_model(model_id=self.modelId)
        self.pathCounts = list(self.model.pathCounts)
        self.multiPathInfoObject = MultipathCombinationInfo(batch_size=self.model.batchSize,
                                                            path_counts=self.pathCounts)

    # Load routing information for the particular model
    def load_model_data(self):
        self.multiPathInfoObject.generate_routing_info(cigt=self.model,
                                                       dataset=self.dataset.testDataTf)
        self.multiPathInfoObject.assert_routing_validity(cigt=self.model)
        self.multiPathInfoObject.assess_accuracy()



