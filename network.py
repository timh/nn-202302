from typing import List
import numpy as np

from layer import Layer

class Network: pass
class Network:
    layers: List[Layer]
    last_input: np.ndarray

    def __init__(self, neurons_input: int, neurons_output: int, num_hidden: int, neurons_hidden: int) -> Network:
        self.layers = list()
        for i in range(num_hidden):
            if i == 0:
                layer = Layer(neurons_hidden, neurons_input)
            else:
                layer = Layer(neurons_hidden, neurons_hidden)
            self.layers.append(layer)
        
        if num_hidden > 0:
            output = Layer(neurons_output, neurons_hidden)
        else:
            output = Layer(neurons_output, neurons_input)
        self.layers.append(output)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.last_input = input
        result = input

        for layer in self.layers:
            result = layer.forward(result)
        return result

    def backward(self, learning_rate: float, loss: float):
        # batch size, num/outputs
        shape = [self.last_input.shape[0], self.layers[-1].biases.shape[0]]
        derivs = np.ones(shape)

        all_derivs = list()
        all_derivs.append(derivs)
        for layer in reversed(self.layers):
            derivs = layer.backward(derivs, learning_rate, loss).T
            all_derivs.append(derivs)
        return derivs

