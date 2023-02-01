import numpy as np

class Layer:
    inputs: np.ndarray             # (batch_size, num_inputs)
    outputs: np.ndarray            # (batch_size, num_neurons)
    dinputs: np.ndarray            # (batch size, num neurons)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise Exception("not implemented")

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        raise Exception("not implemented")

class DenseLayer(Layer):
    weights: np.ndarray            # (num_inputs, num_neurons)
    biases: np.ndarray             # (1, num_neurons)

    dweights: np.ndarray           # (batch_size, num_inputs, num_neurons)
    dbiases: np.ndarray            # (batch_size, 1)
    dinputs: np.ndarray            # (batch_size, num neurons)

    def __init__(self, num_neurons: int, num_weights_per_neuron: int):
        self.weights = 0.1 * np.abs(np.random.randn(num_weights_per_neuron, num_neurons))
        self.biases = np.zeros((1, num_neurons))

        self.inputs = None
        self.outputs = None
        self.dsum = None
        self.dweights = None
        self.dinputs = None

    # returns results for this layer.
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return self.outputs

    # returns derivatives for this layer.
    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

        # result is deriv with regard to the inputs
        return self.dinputs

class ActivationReLU(Layer):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = np.maximum(0., self.inputs)
        return self.outputs

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        self.dinputs = dvalues.copy()
        self.dinputs[self.outputs <= 0] = 0.
        return self.dinputs
