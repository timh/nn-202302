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
        # loss = np.sqrt(loss)
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
    net = Network(neurons_input=1, neurons_hidden=8, num_hidden=2, neurons_output=1)
    inputs = np.array([[x / 10.] for x in range(31)])
    expecteds = np.array([[math.sin(x[0])] for x in inputs])

    print("   inputs: ", array_str(inputs))
    print("expecteds: ", array_str(expecteds))
    print("  weights: ", array_str(net.layers[0].weights))
    print("   biases: ", array_str(net.layers[0].biases))
    results = main(net, inputs, expecteds, 1000, 0.001)
    print("final res: ", results)
    for res in results:
        plt.plot(res)
    plt.show()
