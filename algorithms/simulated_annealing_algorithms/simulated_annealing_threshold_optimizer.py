import numpy as np


class SimulatedAnnealingThresholdOptimizer:
    def __init__(self, network,
                 max_num_of_iterations,
                 annealing_schedule,
                 balance_coefficient,
                 use_weighted_scoring,
                 multipath_score_calculators,
                 verbose):
        self.network = network
        self.maxNumOfIterations = max_num_of_iterations
        self.annealingSchedule = annealing_schedule
        self.balanceCoefficient = balance_coefficient
        self.useWeightedScoring = use_weighted_scoring
        self.multipathScoreCalculators = multipath_score_calculators
        self.verbose = verbose

    def pick_fully_random_state(self):
        threshold_state = {}
        for node in self.network.topologicalSortedNodes:
            if node.isLeaf:
                continue
            child_count = len(self.network.dagObject.children(node=node))
            max_threshold = 1.0 / float(child_count)
            thresholds = max_threshold * np.random.uniform(size=(child_count,))
            threshold_state[node.index] = thresholds
        return threshold_state

    def get_neighbor(self, threshold_state):
        pass

    def calculate_threshold_score(self, threshold_state):
        scores = []
        accuracies = []
        computation_overloads = []
        for scorer in self.multipathScoreCalculators:
            res_method_0, res_method_1 = scorer.calculate_for_threshold(thresholds_dict=threshold_state)
            result = res_method_1 if self.useWeightedScoring else res_method_0
            accuracy_gain = self.balanceCoefficient * result.accuracy
            computation_overload_loss = (1.0 - self.balanceCoefficient) * (result.computationOverload - 1.0)
            score = accuracy_gain - computation_overload_loss
            accuracies.append(result.accuracy)
            computation_overloads.append(result.computationOverload)
            scores.append(score)
        final_score = np.mean(np.array(scores))
        final_accuracy = np.mean(np.array(accuracies))
        final_computation_overload = np.mean(np.array(computation_overloads))
        return final_score, final_accuracy, final_computation_overload

    def get_acceptance(self, curr_score, candidate_score):
        if candidate_score > curr_score:
            return True
        else:
            temperature = self.annealingSchedule.value
            acceptance_threshold = np.exp((candidate_score - curr_score) / temperature)
            rand_val = np.random.uniform(low=0.0, high=1.0)
            if acceptance_threshold >= rand_val:
                return True
            else:
                return False

    def run(self):
        # Get initial state
        curr_state = self.pick_fully_random_state()
        curr_score, curr_accuracy, curr_computation_overload = \
            self.calculate_threshold_score(threshold_state=curr_state)
        for iteration_id in range(self.maxNumOfIterations):
            # Pick a random neighbor
            candidate_state = self.get_neighbor(threshold_state=curr_state)
            candidate_score, candidate_accuracy, candidate_computation_overload = \
                self.calculate_threshold_score(threshold_state=candidate_state)
            accepted = self.get_acceptance(curr_score=curr_score, candidate_score=candidate_score)
            # Report current state
            if self.verbose:
                print("****************Iteration {0}****************".format(iteration_id))
                print("Current Temperature:{0}".format(self.annealingSchedule.value))
                print("Current State:{0}".format(curr_state))
                print("Current Score:{0}".format(curr_score))
                print("Current Accuracy:{0}".format(curr_accuracy))
                print("Current Computation Overload:{0}".format(curr_computation_overload))
                print("Candidate State:{0}".format(candidate_state))
                print("Candidate Score:{0}".format(candidate_score))
                print("Candidate Accuracy:{0}".format(candidate_accuracy))
                print("Candidate Computation Overload:{0}".format(candidate_computation_overload))
            if accepted:
                if self.verbose:
                    print("Candidate accepted.")
                curr_state = candidate_state
                curr_score = candidate_score
                curr_accuracy = candidate_accuracy
                curr_computation_overload = candidate_computation_overload
            else:
                if self.verbose:
                    print("Candidate rejected.")
            if self.verbose:
                print("****************Iteration {0}****************".format(iteration_id))
                self.annealingSchedule.update(iteration=iteration_id)
        return curr_state
