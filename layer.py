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
        self.deriv_relu = np.copy(next_layer_deriv)
        self.deriv_relu[self.deriv_relu < 0] = 0.0
        self.deriv_sum_bias = self.deriv_relu * 1.
        self.deriv_mult_weights_flat = (self.last_input * self.deriv_sum_bias)
        # deriv_mult_weights = deriv_mult_weights_flat.sum(axis=0)
        self.deriv_mult_weights = self.deriv_mult_weights_flat.mean(axis=0)
        # print(f"          deriv_relu: {deriv_relu}")
        # print(f"      deriv_sum_bias: {deriv_sum_bias}")
        # print(f"  deriv_mult_weights: {deriv_mult_weights}")

        self.bias_update = self.deriv_sum_bias * learning_rate * loss
        self.weight_update = self.deriv_mult_weights * learning_rate * loss
        if len(self.bias_update.shape) == 2:
            self.bias_update = np.sum(self.bias_update, axis=0)
            self.weight_update = np.sum(self.weight_update, axis=0)
        self.biases -= self.bias_update
        self.weights -= self.weight_update

        res = self.deriv_mult_weights_flat.T
        # print(f"         bias_update: {bias_update}")
        # print(f"       weight_update: {weight_update}")
        # print(f"             weights: {self.weights}")
        # print(f"              biases: {self.biases}")
        # print(f"               return {res}")
        return res

