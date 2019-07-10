import numpy as np

from algorithms.threshold_optimization_algorithms.threshold_optimizer import ThresholdOptimizer


class SimulatedAnnealingThresholdOptimizer(ThresholdOptimizer):
    def __init__(self, network, max_num_of_iterations, annealing_schedule, balance_coefficient, use_weighted_scoring,
                 multipath_score_calculators, verbose):
        super().__init__(network, multipath_score_calculators, balance_coefficient, use_weighted_scoring, verbose)
        self.maxNumOfIterations = max_num_of_iterations
        self.annealingSchedule = annealing_schedule

    def get_neighbor(self, threshold_state):
        pass

    def get_acceptance(self, curr_score, candidate_score):
        if candidate_score > curr_score:
            return True
        else:
            temperature = self.annealingSchedule.value
            acceptance_threshold = np.exp((candidate_score - curr_score) / temperature)
            if self.verbose:
                print("Temperature:{0}".format(temperature))
                print("Acceptance Threshold:{0}".format(acceptance_threshold))
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
