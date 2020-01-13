import tensorflow as tf


class PolicyGradientsNetwork:
    def __init__(self, action_spaces, state_shapes, l2_lambda):
        self.actionSpaces = action_spaces
        self.l2Lambda = l2_lambda
        self.paramL2Norms = {}
        self.l2Loss = None
        self.trajectoryMaxLength = len(action_spaces)
        assert len(state_shapes) == self.trajectoryMaxLength
        self.stateShapes = state_shapes
        self.inputs = []
        self.policies = []
        self.rewards = []

    def build_policy_networks(self):
        pass

    def state_transition(self, history):
        pass

    def build_policy_gradient_loss(self):
        pass

    def build_network(self):
        # State inputs and reward inputs
        for t in range(self.trajectoryMaxLength):
            # States
            input_shape = [None]
            input_shape.extend(self.stateShapes[t])
            state_input = tf.placeholder(dtype=tf.float32, shape=input_shape, name="inputs_{0}".format(t))
            self.inputs.append(state_input)
            # Rewards
            reward_shape = [None, len(self.actionSpaces[t])]
            reward_input = tf.placeholder(dtype=tf.float32, shape=reward_shape, name="rewards_{0}".format(t))
            self.rewards.append(reward_input)
        # Build policy generating networks; self.policies are filled.
        self.build_policy_networks()

    def get_l2_loss(self):
        # L2 Loss
        tvars = tf.trainable_variables()
        self.l2Loss = tf.constant(0.0)
        for tv in tvars:
            if 'kernel' in tv.name:
                self.l2Loss += self.l2Lambda * tf.nn.l2_loss(tv)
            self.paramL2Norms[tv.name] = tf.nn.l2_loss(tv)
