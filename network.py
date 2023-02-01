from typing import List
import numpy as np

from layer import Layer, DenseLayer, ActivationReLU

class Network: pass
class Network:
    layers: List[Layer]
    inputs: np.ndarray

    def __init__(self, num_inputs: int, neurons_output: int, layers_hidden: int, neurons_hidden: int) -> Network:
        self.layers = list()
        for i in range(layers_hidden):
            if i == 0:
                layer = DenseLayer(neurons_hidden, num_inputs)
            else:
                layer = DenseLayer(neurons_hidden, neurons_hidden)
            self.layers.append(layer)
            self.layers.append(ActivationReLU())
        
        if layers_hidden > 0:
            output = DenseLayer(neurons_output, neurons_hidden)
        else:
            output = DenseLayer(neurons_output, num_inputs)
        self.layers.append(output)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs

        result = self.inputs
        for layer in self.layers:
            result = layer.forward(result)

        return result

    def backward(self, learning_rate: float, loss: float):
        # batch size, num/outputs
        last_layer = self.layers[-1]
        last_outputs = last_layer.outputs
        derivs = np.ones(last_outputs.shape)
        derivs *= loss

        all_derivs = list()
        all_derivs.append(derivs)
        for layer in reversed(self.layers):
            derivs = layer.backward(derivs)
            all_derivs.append(derivs)
        
        return derivs
    
    def update(self, learning_rate: float):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.biases -= layer.dbiases * learning_rate
                layer.weights -= layer.dweights * learning_rate



