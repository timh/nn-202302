from typing import List
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageFont, ImageDraw

from layer import Layer
from network import Network
import html_util

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

        res = net.forward(rotated_inputs).T[0]

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

        if step % 1 == 0:
            with open(f"step-{step}.html", "w") as out:
                html_util.draw_step(net, out)

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
