from typing import List
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageFont, ImageDraw

from layer import Layer
from network import Network
import html_util

def array_str(array: np.ndarray) -> str:
    if len(array.shape) > 1:
        child_strs = [array_str(child) for child in array]
        child_strs = ", ".join(child_strs)
    else:
        child_strs = [format(v, ".4f") for v in array]
        child_strs = ", ".join(child_strs)
    return f"[{child_strs}]"

def main(net: Network, inputs: np.ndarray, expected: np.ndarray, steps: int, learning_rate: float):
    results = list()

    out = open(f"steps.html", "w")
    print("<html>", file=out)
    print("<head>", file=out)
    print("<link rel=\"stylesheet\" href=\"net.css\"></link>", file=out)
    print("</head>", file=out)
    print("<body>", file=out)

    for step in range(steps):
        print(f"step {step}")

        rotated_inputs = inputs
        res = net.forward(rotated_inputs)

        # real_loss = res - expecteds
        real_loss = res - expecteds
        loss = real_loss * real_loss
        loss = np.sum(loss)
        loss = np.sqrt(loss)
        if np.sum(real_loss) < 0:
            loss = -loss
        
        res_str = array_str(res)
        exp_str = array_str(expected)
        print(f"              res {res_str}\n"
              f"         expected {exp_str}\n"
              f"        real_loss {array_str(real_loss)}\n"
              f"             loss {loss}")
        
        derivs = net.backward(learning_rate, loss)
        # print(f"  derivs {derivs}")
        results.append(res)

        html_util.draw_step(net, out)
    
    print("</body>", file=out)
    print("</html>", file=out)

    return results

if __name__ == "__main__":
    # net = Network(1, 1, 2, 6)

    inputs = np.array([[1.0, -2.0, 3.0], [2, 3, 1], [3, 4, 5], [2,2,2]])
    expecteds = np.array([x * y * z for x, y, z in inputs])

    # net = Network(3, 1, 0, 0)
    # net.layers[0].weights = np.array([
    #     [-3.0, -1.0, 2.0]
    # ]).T
    # net.layers[0].biases = np.array([
    #     [6.0]
    # ]).T
    net = Network(neurons_input=1, neurons_hidden=2, num_hidden=1, neurons_output=3)
    # inputs = np.array([[4.1], [4.2], [4.3], [4.4]])
    # expecteds = np.array([[x[0], x[0], x[0]] for x in inputs])
    inputs = np.array([[1], [2], [3], [4]])
    expecteds = np.array([[x[0], x[0], x[0]] for x in inputs])

    print("   inputs: ", array_str(inputs))
    print("expecteds: ", array_str(expecteds))
    print("  weights: ", array_str(net.layers[0].weights))
    print("   biases: ", array_str(net.layers[0].biases))
    results = main(net, inputs, expecteds, 100, 0.1)
    for res in results:
        if len(res.shape) > 1:
            for subres in res:
                plt.plot(subres)
        else:
            plt.plot(res)
    plt.show()
