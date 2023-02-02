from typing import List
import numpy as np

from layer import Layer, DenseLayer, ActivationReLU
from loss import Loss

class Network:
    layers: List[Layer]
    inputs: np.ndarray
    loss_obj: Loss

    def __init__(self, num_inputs: int, neurons_output: int, layers_hidden: int, neurons_hidden: int, loss_obj: Loss) -> 'Network':
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
        self.loss_obj = loss_obj

    def forward(self, inputs: np.ndarray, expected_truth: np.ndarray) -> np.ndarray:
        self.inputs = inputs

        result = self.inputs
        for layer in self.layers:
            result = layer.forward(result)
        
        if self.loss_obj is not None:
            self.loss_obj.get_loss(result, expected_truth)
            result = self.loss_obj.output

        self.outputs = result
        return result

    def backward(self, expected_truth: np.ndarray):
        dvalues = self.loss_obj.backward(self.outputs, expected_truth)
        # # batch size, num/outputs
        # last_layer = self.layers[-1]
        # last_outputs = last_layer.outputs
        # dvalues = np.ones(last_outputs.shape)
        # dvalues *= self.loss_obj.loss

        all_derivs = list()
        all_derivs.append(dvalues)
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)
            all_derivs.append(dvalues)
        
        return dvalues
    
    def update_params(self, learning_rate: float):
        for layer in self.layers:
            layer.update_params(learning_rate)
