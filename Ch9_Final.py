import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer:
    inputs: np.ndarray   # (Inputs) or (Batch, Inputs)
    outputs: np.ndarray  # (Outputs) or (Batch, Outputs)
    dinputs: np.ndarray  # (Outputs, Neurons) HACK is this right?

    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.dinputs = None

    # Forward pass
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise Exception("not implemented")

    # Backward pass
    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        raise Exception("not implemented")


# Dense layer
class Layer_Dense(Layer):
    weights: np.ndarray  # (Inputs, Neurons)
    biases: np.ndarray   # (1, Neurons)

    dweights: np.ndarray # (Inputs, Neurons)
    dbiases: np.ndarray  # (1, Neurons)


    # Layer initialization
    def __init__(self, n_inputs: int, n_neurons: int):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.dweights = None
        self.dbiases = None

    # Forward pass
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    # Backward pass
    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs


# ReLU activation
class Activation_ReLU(Layer):

    # Forward pass
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)
        return self.output

    # Backward pass
    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

        return self.dinputs


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


# Common loss class
class Loss:
    def forward(self, output: np.ndarray, expected_truth: np.ndarray) -> np.ndarray:
        raise Exception("not implemented")

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output: np.ndarray, expected_truth: np.ndarray) -> float:
        # Calculate sample losses
        sample_losses = self.forward(output, expected_truth)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, output: np.ndarray, expected_truth: np.ndarray) -> np.ndarray:

        # Number of samples in a batch
        samples = len(output)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(output, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(expected_truth.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                expected_truth
            ]


        # Mask values - only for one-hot encoded labels
        elif len(expected_truth.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * expected_truth,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues: np.ndarray, expected_truth: np.ndarray) -> np.ndarray:

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(expected_truth.shape) == 1:
            expected_truth = np.eye(labels)[expected_truth]

        # Calculate gradient
        self.dinputs = -expected_truth / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

        return self.dinputs


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs: np.ndarray, expected_truth: np.ndarray) -> np.ndarray:
        # Output layer's activation function
        self.activation.forward(inputs)

        # Set the output
        self.output = self.activation.output

        # Calculate and return loss value
        return self.loss.calculate(self.output, expected_truth)


    # Backward pass
    def backward(self, dvalues: np.ndarray, expected_truth: np.ndarray) -> np.ndarray:
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(expected_truth.shape) == 2:
            expected_truth = np.argmax(expected_truth, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), expected_truth] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

        return self.dinputs


if __name__ == "__main__":
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)

    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 3)

    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()

    # Create second Dense layer with 3 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = Layer_Dense(3, 3)

    # Create Softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)

    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y)
    # Let's see output of the first few samples:
    print("loss_activation.output[:5]\n", loss_activation.output[:5])

    # Print loss value
    print("loss:", loss)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    # Print accuracy
    print(" acc:", accuracy)

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Print gradients
    print("dense1.dweights:\n", dense1.dweights)
    print("dense1.dbiases:\n", dense1.dbiases)
    print("dense2.dweights:\n", dense2.dweights)
    print("dense2.dbiases:\n", dense2.dbiases)
