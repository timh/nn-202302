import numpy as np

class Layer: pass
class Layer:
    weights: np.ndarray            # N(um neurons) x P(previous layer neurons)
    biases: np.ndarray             # N(um neurons)
    save_weights: np.ndarray
    save_biases: np.ndarray
    last_result: np.ndarray        # N(um neurons)

    deriv_sum_bias: np.ndarray     # N(um neurons)
    deriv_mult_weights: np.ndarray # N(um neurons) * M(num neurons)
    deriv_relu: np.ndarray         # N(um neurons)

    def __init__(self, num_neurons: int, num_weights_per_neuron: int):
        self.weights = 0.1 * np.abs(np.random.randn(num_weights_per_neuron, num_neurons))
        self.biases = np.zeros((1, num_neurons))

        self.last_result = None
        self.deriv_sum_bias = None
        self.deriv_mult_weights = None
        self.deriv_relu = None

    # returns results for this layer.
    def forward(self, input: np.ndarray) -> np.array:
        # with relu.
        self.last_sum_no_bias = np.dot(input, self.weights)
        self.last_sum = self.last_sum_no_bias + self.biases
        self.last_result = np.maximum(0., self.last_sum)
        self.last_input = np.array(input)
        return self.last_result

    # returns derivatives for this layer.
    def backward(self, next_layer_deriv: np.ndarray, learning_rate: float, loss: float) -> np.ndarray:
        self.save_weights = np.copy(self.weights)
        self.save_biases = np.copy(self.biases)

        # print(f"\nbackward: {self.weights.shape}")
        # print(f"    next_layer_deriv: {next_layer_deriv}")
        self.deriv_relu = next_layer_deriv.copy()
        self.deriv_relu[self.last_result <= 0] = 0.

        self.deriv_weights = np.dot(self.last_input.T, self.deriv_relu)
        self.deriv_inputs = np.dot(self.deriv_relu, self.weights.T)
        self.deriv_biases = np.sum(self.deriv_relu, axis=0, keepdims=True)

        self.bias_update = self.deriv_biases * learning_rate
        self.weight_update = self.deriv_weights * learning_rate
        self.biases -= self.bias_update
        self.weights -= self.weight_update

        # result is deriv with regard to the inputs
        res = self.deriv_inputs

        # print(f"         bias_update: {bias_update}")
        # print(f"       weight_update: {weight_update}")
        # print(f"             weights: {self.weights}")
        # print(f"              biases: {self.biases}")
        # print(f"               return {res}")
        return res

