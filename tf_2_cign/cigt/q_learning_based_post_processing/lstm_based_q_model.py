import tensorflow as tf


class LstmBasedQModel(tf.keras.Model):
    def __init__(self, path_counts, input_dimension, lstm_layer_dimensions, dropout_ratio, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathCounts = path_counts
        self.blockCount = len(self.pathCounts) - 1
        self.dropOutRatio = dropout_ratio
        self.lstmLayerDimensions = lstm_layer_dimensions
        self.inputFullyConnectedLayers = []
        self.outputFullyConnectedLayers = []
        self.lstmLayers = []
        for block_id in range(self.blockCount):
            input_layer = tf.keras.layers.Dense(units=input_dimension, activation="relu")
            self.inputFullyConnectedLayers.append(input_layer)
            output_layer = tf.keras.layers.Dense(units=2, activation=None)
            self.outputFullyConnectedLayers.append(output_layer)

        for layer_id, lstm_dim in enumerate(self.lstmLayerDimensions):
            lstm_layer = tf.keras.layers.LSTM(lstm_dim, return_sequences=True, dropout=dropout_ratio)
            self.lstmLayers.append(lstm_layer)

    def call(self, inputs, training=None, mask=None):
        lstm_inputs = []
        for block_id in range(self.blockCount):
            block_input = self.inputFullyConnectedLayers[block_id](inputs[block_id])
            lstm_inputs.append(block_input)
        # Convert into a [batch, timesteps, feature] shaped LSTM input.
        lstm_input = tf.stack(lstm_inputs, axis=1)
        # Run through lstm layers
        x = lstm_input
        for lstm_layer in self.lstmLayers:
            x = lstm_layer(x)
        # Convert the lstm outputs to regression outputs
        regression_outputs = []
        for block_id in range(self.blockCount):
            lstm_output = x[:, block_id, :]
            lstm_output = self.outputFullyConnectedLayers[block_id](lstm_output)
            regression_outputs.append(lstm_output)
        return regression_outputs







