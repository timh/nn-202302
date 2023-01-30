from typing import List
import numpy as np

class Layer: pass

class Layer:
    weights: np.ndarray            # N(um neurons) x P(previous layer neurons)
    biases: np.ndarray             # N(um neurons)
    last_result: np.ndarray        # N(um neurons)

    deriv_sum_bias: np.ndarray     # N(um neurons)
    deriv_mult_weights: np.ndarray # N(um neurons) * M(num neurons)
    deriv_relu: np.ndarray         # N(um neurons)

    def __init__(self, num_neurons: int, num_weights_per_neuron: int):
        self.weights = np.random.randn(num_weights_per_neuron, num_neurons)
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
        # print(f"\nbackward: {self.weights.shape}")
        # print(f"    next_layer_deriv: {next_layer_deriv}")
        deriv_relu = np.copy(next_layer_deriv)
        deriv_relu[deriv_relu < 0] = 0.0
        deriv_sum_bias = deriv_relu * 1.
        deriv_mult_weights_flat = (self.last_input * deriv_sum_bias)
        deriv_mult_weights = deriv_mult_weights_flat.sum(axis=0)
        # print(f"          deriv_relu: {deriv_relu}")
        # print(f"      deriv_sum_bias: {deriv_sum_bias}")
        # print(f"  deriv_mult_weights: {deriv_mult_weights}")

        bias_update = deriv_sum_bias * learning_rate * loss
        weight_update = deriv_mult_weights * learning_rate * loss
        if len(bias_update.shape) == 2:
            bias_update = np.sum(bias_update, axis=0)
            weight_update = np.sum(weight_update, axis=0)
        self.biases -= bias_update
        self.weights -= weight_update

        res = deriv_mult_weights_flat.T
        # print(f"         bias_update: {bias_update}")
        # print(f"       weight_update: {weight_update}")
        # print(f"             weights: {self.weights}")
        # print(f"              biases: {self.biases}")
        # print(f"               return {res}")
        return res

class Network: pass
class Network:
    layers: List[Layer]
    input: np.ndarray
    output: Layer
    hidden: List[Layer]
    last_input: np.ndarray

    def __init__(self, neurons_input: int, neurons_output: int, num_hidden: int, neurons_hidden: int) -> Network:
        self.input = np.zeros((neurons_input, ))
        self.output = Layer(neurons_output, neurons_hidden)
        self.hidden = list()
        for i in range(num_hidden):
            if i == 0:
                layer = Layer(neurons_hidden, neurons_input)
            else:
                layer = Layer(neurons_hidden, neurons_hidden)
            self.hidden.append(layer)
    
    def forward(self, input: np.ndarray) -> np.array:
        self.last_input = input
        next_input = input
        for layer in self.hidden:
            next_input = layer.forward(next_input)
        result = self.output.forward(next_input)
        return result

    def backward(self, learning_rate: float, loss: float):
        all_derivs = list()
        derivs = np.ones_like(self.last_input)
        all_derivs.append(derivs)
        derivs = self.output.backward(derivs, learning_rate, loss).T
        all_derivs.append(derivs)
        for layer in reversed(self.hidden):
            derivs = layer.backward(derivs, learning_rate, loss).T
            all_derivs.append(derivs)
        return derivs

if __name__ == "__main__":
    net = Network(1, 1, 1, 3)

    inputs = [1, 2, 3, 4, 5]
    expecteds = [x*2 for x in inputs]

    for step in range(100):
        print(f"step {step}:")
        if False:
            for i, (one_input, expected) in enumerate(zip(inputs, expecteds)):
                res = net.forward(np.array([one_input]))

                loss = np.sum(res) - np.sum(expected)
                print(f"{step}: f({one_input}) = {res}, loss = {loss}")
                derivs = net.backward(0.01, loss)
        else:
            rotated_inputs = np.array(inputs)[np.newaxis].T
            res = net.forward(rotated_inputs)
            loss = np.sum(res - expecteds)
            print(f"       res {res.T[0]}\n"
                  f"  expected {expecteds}\n"
                  f"      loss {loss}")
            derivs = net.backward(0.000001, loss)
            # print(f"  derivs {derivs}")
    
    print(f"final result: {res}")
