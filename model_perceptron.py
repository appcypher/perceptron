import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, num_iterations=10_000, input_dim=2):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.input_dim = input_dim
        self.weights = np.zeros((input_dim,))
        self.activation_threshold = -1
        self.bias = 0.0
        self.accuracy = 0.0

    def train(self, samples, labels):
        """
        Trains the model using the delta rule.
        """
        assert samples.shape[1] == self.input_dim, "Training samples must have consistent length"
        assert len([x != 0 and x != 1 for x in labels]) > 0, "Training labels must either be 1 or 0"
        assert len(samples) == len(labels), "Lengths of training samples and labels must match"

        for _ in range(self.num_iterations):
            for i, sample in enumerate(samples):
                prediction = np.dot(self.weights, sample) + self.bias
                prediction = self.unit_step(prediction)
                update = self.learning_rate * (labels[i] - prediction)
                self.weights += update * sample
                self.bias += update

    def test(self, samples, expected_outputs):
        sample_size, input_dim = samples.shape
        expected_output_size = len(expected_outputs)

        assert input_dim == self.input_dim, "Test input must have consistent length"
        assert sample_size == expected_output_size, "Lengths of test samples and expected outputs must match"

        predictions = np.dot(samples, self.weights) + self.bias
        predictions = self.unit_step(predictions)

        self.accuracy = np.sum(predictions == expected_outputs) / expected_output_size

        return predictions

    def unit_step(self, x):
        return np.where(x > self.activation_threshold, 1, 0)






