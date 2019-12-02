import numpy as npimport tensorflow as tffrom sklearn.preprocessing import StandardScalerfrom algorithms.bayesian_optimization import BayesianOptimizerfrom algorithms.threshold_optimization_algorithms.routing_weight_calculator import RoutingWeightCalculatorfrom auxillary.general_utility_funcs import UtilityFuncsfrom simple_tf.global_params import GlobalConstantsfeature_names = ["posterior_probs"]class RoutingWeightBayesianOptimizer(RoutingWeightCalculator):    def __init__(self, network, validation_routing_matrix, test_routing_matrix, validation_data, test_data,                 min_weight, max_weight, best_val_accuracy, best_test_accuracy):        super().__init__(network, validation_routing_matrix, test_routing_matrix, validation_data, test_data)        self.routingCombinations = UtilityFuncs.get_cartesian_product(list_of_lists=[[0, 1]] * len(self.leafNodes))        self.routingCombinations = [np.array(route_vec) for route_vec in self.routingCombinations]        self.validation_X, self.validation_Y, self.test_X, self.test_Y = \            self.build_data_sets(selected_features=GlobalConstants.SELECTED_FEATURES_FOR_WEIGHT_REGRESSION)        self.currRouteVector = None        self.minWeight = min_weight        self.maxWeight = max_weight        self.bestValAccuracy = best_val_accuracy        self.bestTestAccuracy = best_test_accuracy        self.bounds = [(self.minWeight, self.maxWeight)] * len(self.leafNodes)        self.optimalWeightsDict = {}    def random_state_sampler(self):        valid_leaves_count = np.sum(self.currRouteVector)        random_state = np.random.uniform(low=self.minWeight, high=self.maxWeight, size=(valid_leaves_count,))        return random_state    def calculate_threshold_score(self, x, sparse_posteriors, labels):        weights = np.copy(self.currRouteVector).astype(np.float32)        weights[np.nonzero(weights)] = x        weighted_posteriors = sparse_posteriors * np.expand_dims(np.expand_dims(weights, axis=0), axis=0)        final_posteriors = np.sum(weighted_posteriors, axis=2)        predicted_labels = np.argmax(final_posteriors, axis=1)        score_accuracy = np.sum(predicted_labels == labels).astype(np.float32) / labels.shape[0]        return score_accuracy    def calculate_threshold_score_val(self, x):        element_wise_compliance = self.validationRoutingMatrix == self.currRouteVector        indicator_vector = np.all(element_wise_compliance, axis=1)        if np.sum(indicator_vector) == 0:            return 0.0        sparse_posteriors = self.validationSparsePosteriors[indicator_vector, :]        labels = self.validationData.labelList[indicator_vector]        score_accuracy = self.calculate_threshold_score(x=x, sparse_posteriors=sparse_posteriors, labels=labels)        return score_accuracy, None    def calculate_threshold_score_test(self, x):        element_wise_compliance = self.testRoutingMatrix == self.currRouteVector        indicator_vector = np.all(element_wise_compliance, axis=1)        if np.sum(indicator_vector) == 0:            return 0.0        sparse_posteriors = self.testSparsePosteriors[indicator_vector, :]        labels = self.testData.labelList[indicator_vector]        score_accuracy = self.calculate_threshold_score(x=x, sparse_posteriors=sparse_posteriors, labels=labels)        return score_accuracy, None    def test_uniform_weights(self):        route_performances_validation = {}        route_performances_test = {}        for route_vec in self.routingCombinations:            route_vector = np.array(route_vec)            self.currRouteVector = route_vector            if np.sum(route_vector) < 1:                continue            route_vector_as_tuple = tuple(route_vector.tolist())            element_wise_compliance = self.validationRoutingMatrix == self.currRouteVector            indicator_vector_val = np.all(element_wise_compliance, axis=1)            element_wise_compliance = self.testRoutingMatrix == self.currRouteVector            indicator_vector_test = np.all(element_wise_compliance, axis=1)            res_val = self.calculate_threshold_score_val(                x=(1.0 / np.sum(route_vector)) * np.ones(shape=(int(np.sum(route_vector)),)))            res_test = self.calculate_threshold_score_test(                x=(1.0 / np.sum(route_vector)) * np.ones(shape=(int(np.sum(route_vector)),)))            route_performances_validation[route_vector_as_tuple] = (res_val[0], np.sum(indicator_vector_val))            route_performances_test[route_vector_as_tuple] = (res_test[0], np.sum(indicator_vector_test))        assert np.allclose(self.bestValAccuracy,                           sum([tpl[0] * tpl[1] for tpl in route_performances_validation.values()]) /                           self.validationData.labelList.shape[0])        assert np.allclose(self.bestTestAccuracy,                           sum([tpl[0] * tpl[1] for tpl in route_performances_test.values()]) /                           self.testData.labelList.shape[0])        print(route_performances_validation)        print(route_performances_test)        print("X")    def run(self):        self.test_uniform_weights()        for route_vec in self.routingCombinations:            route_vector = np.array(route_vec)            self.currRouteVector = route_vector            if np.sum(route_vector) <= 1:                continue            route_vector_as_tuple = tuple(route_vector.tolist())            element_wise_compliance = self.validationRoutingMatrix == self.currRouteVector            indicator_vector_val = np.all(element_wise_compliance, axis=1)            if np.sum(indicator_vector_val) == 0:                self.optimalWeightsDict[route_vector_as_tuple] = (1.0 / np.sum(route_vector)) * route_vector                continue            bayesian_optimizer = BayesianOptimizer(xi=0.01, bounds=self.bounds[0:int(np.sum(route_vector))],                                                   random_state_func=self.random_state_sampler,                                                   score_func_val=self.calculate_threshold_score_val,                                                   score_func_test=self.calculate_threshold_score_test)            best_result, all_results = bayesian_optimizer.run(max_iterations=100, initial_sample_count=100)        # print("X")