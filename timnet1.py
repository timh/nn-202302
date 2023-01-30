from typing import List
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageFont, ImageDraw
from fonts.ttf import Roboto

from image_util import draw_multiline

font = ImageFont.truetype(Roboto, 10)

class Layer: pass
class Layer:
    weights: np.ndarray            # N(um neurons) x P(previous layer neurons)
    biases: np.ndarray             # N(um neurons)
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

    def forward(self, input: np.ndarray, image: Image.Image = None) -> np.array:
        self.last_input = input
        result = input

        if image is not None:
            draw = ImageDraw.Draw(image)

            x = 0
            for input_one in reversed(input):
                one_input_array = []
                for value in input_one:
                    one_input_array.append(f"{value:10.4f}")
                one_input_str = "\n".join(one_input_array)
                xy_input = draw_multiline(one_input_str, (x, 0), "blue", image, draw, font)
                x = xy_input[0]
            
            x += 20

        for lidx, layer in enumerate(self.layers):
            if image is not None:
                num_neurons = layer.weights.shape[1]
                maxx = x
                neuron_tops = list()
                neuron_bots = list()
                y = 0
                border = 2
                for nidx in range(num_neurons):
                    weights = [format(v, ".4f") for v in layer.weights.T[nidx]]
                    bias = format(layer.biases.T[nidx][0], ".4f")

                    weight_bbox = draw_multiline(weights, (x + border, y + border), "black", image, draw, font)
                    biasx = (x + weight_bbox[0]) / 2
                    biasy = weight_bbox[1]
                    bias_bbox = draw_multiline(bias, (biasx, biasy), "black", image, draw, font)

                    top = y
                    bot = bias_bbox[1] + border
                    draw.rectangle((x, y, bias_bbox[0], bot), outline="black")

                    neuron_tops.append(top)
                    neuron_bots.append(bot)
                    y = bot + 100
                    maxx = max(maxx, bias_bbox[0])
                
                x = maxx + 10

            result = layer.forward(result)
            if image is not None:
                maxx = x
                for iidx, one_res in enumerate(result.T):
                    y = neuron_tops[iidx]
                    one_res_str = "\n".join([format(v, ".4f") for v in one_res])
                    res_bbox = draw_multiline(one_res_str, (x, y), "red", image, draw, font)
                    maxx = max(maxx, x + res_bbox[0])
            
                x = maxx + 20


        return result

    def backward(self, learning_rate: float, loss: float, image: Image = None):
        # batch size, num/outputs
        shape = [self.last_input.shape[0], self.layers[-1].biases.shape[0]]
        derivs = np.ones(shape)

        all_derivs = list()
        all_derivs.append(derivs)
        for layer in reversed(self.layers):
            derivs = layer.backward(derivs, learning_rate, loss).T
            all_derivs.append(derivs)
        return derivs

def main(net: Network, inputs: np.ndarray, expected: np.ndarray, steps: int):
    results = list()

    for step in range(steps):
        print(f"step {step}:")
        for layer in net.layers:
            weights = ""
            biases = ""
            for neuron in layer.weights.T:
                weights += "\n["
                for weight in neuron:
                    weights += format(weight, "9.4f") + " "
                weights += "]"
            for neuron in layer.biases.T:
                biases += "\n["
                for bias in neuron:
                    biases += format(bias, "9.4f") + " "
                biases += "]"

            print(f"  weights {layer.weights.T.shape} = {weights}")
            print(f"   biases {layer.biases.T.shape} = {biases}")
        
        # rotated_inputs = np.array(inputs)[np.newaxis].T
        rotated_inputs = inputs

        image = None
        if step % 1 == 0:
            image = Image.new(mode="RGB", size=(1024, 512), color="white")
        res = net.forward(rotated_inputs, image).T[0]
        if image is not None:
            image.save(f"step-{step}.png")

        # real_loss = res - expecteds
        real_loss = res - expecteds
        loss = real_loss * real_loss
        loss = np.mean(loss)
        if np.sum(real_loss) < 0:
            loss = -loss

        
        res_str = "[" + " ".join([format(v, ".4f") for v in res]) + "]"
        exp_str = "[" + " ".join([format(v, ".4f") for v in expecteds]) + "]"
        print(f"              res {res_str}\n"
              f"         expected {exp_str}\n"
              f"        real_loss {real_loss}\n"
              f"             loss {loss}")
        
        derivs = net.backward(0.1, loss)
        # print(f"  derivs {derivs}")
        results.append(res)
    

    return results

if __name__ == "__main__":
    # net = Network(1, 1, 2, 6)

    # inputs = list(range(8))
    # expecteds = [math.sin(x * math.pi/8) for x in inputs]
    inputs = np.array([[1.0, -2.0, 3.0]])
    expecteds = np.array([1])

    net = Network(3, 1, 0, 0)
    net.layers[0].weights = np.array([
        [-3.0, -1.0, 2.0]
    ]).T
    net.layers[0].biases = np.array([
        [6.0]
    ]).T
    print(f"weights {net.layers[0].weights}")
    print(f" biases {net.layers[0].biases}")
    results = main(net, inputs, expecteds, 10)
    # for res in results:
    #     plt.plot(res)
    # plt.show()
