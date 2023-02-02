import numpy as np

from layer import Activation_Softmax

##
# copied from NNFS chapter 9
##
# Common loss class
class Loss:
    loss: float

    def get_loss(self, output: np.ndarray, expected_truth: np.ndarray) -> float:
        raise Exception("not implemented")

    def backward(self, output: np.ndarray, expected_truth: np.ndarray) -> np.ndarray:
        raise Exception("not implemented")

    # Calculates the data and regularization losses given model output and ground truth values
    def calculate(self, output: np.ndarray, expected_truth: np.ndarray) -> float:
        # Calculate sample losses
        sample_losses = self.get_loss(output, expected_truth)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def get_loss(self, output: np.ndarray, expected_truth: np.ndarray) -> float:
        # Number of samples in a batch
        samples = len(output)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        expected_clipped = np.clip(output, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(expected_truth.shape) == 1:
            correct_confidences = expected_clipped[
                range(samples),
                expected_truth
            ]

        # Mask values - only for one-hot encoded labels
        elif len(expected_truth.shape) == 2:
            correct_confidences = np.sum(
                expected_clipped * expected_truth,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        self.loss = negative_log_likelihoods
        return self.loss

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


# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy(Loss):
    activation: Activation_Softmax
    loss_obj: Loss_CategoricalCrossentropy

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss_obj = Loss_CategoricalCrossentropy()

    # Forward pass
    def get_loss(self, inputs: np.ndarray, expected_truth: np.ndarray) -> float:
        # Output layer's activation function
        self.output = self.activation.forward(inputs)

        # Calculate and return loss value
        self.loss = self.loss_obj.calculate(self.output, expected_truth)
        return self.loss

    # Backward pass
    def backward(self, dvalues: np.ndarray, expected_truth: np.ndarray) -> np.ndarray:
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded, turn them into discrete values
        if len(expected_truth.shape) == 2:
            expected_truth = np.argmax(expected_truth, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()

        # Calculate gradient
        self.dinputs[range(samples), expected_truth] -= 1

        # Normalize gradient
        self.dinputs = self.dinputs / samples

        return self.dinputs
