import numpy as np

class Layer:
    inputs: np.ndarray             # (batch_size, num_inputs)
    outputs: np.ndarray            # (batch_size, num_neurons)
    dinputs: np.ndarray            # (batch size, num neurons)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise Exception("not implemented")

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        raise Exception("not implemented")
    
    def update_params(self, learning_rate: float):
        # do nothing.
        pass

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
    
    def update_params(self, learning_rate: float):
        self.biases -= self.dbiases * learning_rate
        self.weights -= self.dweights * learning_rate


class ActivationReLU(Layer):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = np.maximum(0., self.inputs)
        return self.outputs

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        self.dinputs = dvalues.copy()
        self.dinputs[self.outputs <= 0] = 0.
        return self.dinputs

# copied from NNFS chapter 9
# Softmax activation
class Activation_Softmax(Layer):
    # Forward pass
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
        return self.output

    # Backward pass
    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        
        return self.dinputs
