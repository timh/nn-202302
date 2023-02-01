from typing import List
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageFont, ImageDraw

from layer import Layer, DenseLayer
from network import Network
from Ch9_Final import Activation_Softmax_Loss_CategoricalCrossentropy as ASLCC
import html_util

def array_str(array: np.ndarray) -> str:
    if len(array.shape) > 1:
        child_strs = [array_str(child) for child in array]
        child_strs = ", ".join(child_strs)
    else:
        child_strs = [format(v, ".4f") for v in array]
        child_strs = ", ".join(child_strs)
    return f"[{child_strs}]"

def main(net: Network, inputs: np.ndarray, expected: np.ndarray, steps: int, learning_rate: float) -> np.ndarray:
    loss_values: List[float] = list()

    out = open(f"steps.html", "w")
    print("<html>", file=out)
    print("<head>", file=out)
    print("<link rel=\"stylesheet\" href=\"net.css\"></link>", file=out)
    print("</head>", file=out)
    print("<body>", file=out)

    for step in range(steps):

        loss_cc = ASLCC()
        rotated_inputs = inputs
        outputs = net.forward(rotated_inputs)
        loss = loss_cc.forward(outputs, expected)
        loss_values.append(loss)

        # outputs_str = array_str(outputs)
        outputs_str = array_str(loss_cc.output)
        exp_str = array_str(expected)
        if step == steps - 1:
            print(f"step {step}")
            print(f" outputs {outputs_str}\n"
                  f"expected {exp_str}\n"
                  f"    loss {loss:.4f}")

        dvalues = loss_cc.output
        dvalues = loss_cc.backward(dvalues, expected)
        for lidx, layer in enumerate(reversed(net.layers)):
            lidx = len(net.layers) - lidx - 1
            dvalues = layer.backward(dvalues)
            # if isinstance(layer, DenseLayer):
            #     print(f"  {lidx} weights", array_str(layer.weights))
            #     print(f"  {lidx}  biases", array_str(layer.biases))
            #     print(f"  {lidx} outputs", array_str(layer.outputs))

        
        net.update(learning_rate)
        # html_util.draw_step(net, out)
        # print()
    
    print("</body>", file=out)
    print("</html>", file=out)

    return loss_values

if __name__ == "__main__":
    net = Network(num_inputs=1, neurons_hidden=10, layers_hidden=2, neurons_output=2)
    input_vals = [x if x % 2 == 0 else -x for x in range(30)]
    inputs = np.array([[x] for x in input_vals])
    expecteds = np.array([[1. if x >= 0 else 0., 1. if x < 0 else 0.] for x in input_vals])

    print("   inputs: ", array_str(inputs))
    print("expecteds: ", array_str(expecteds))
    print("  weights: ", array_str(net.layers[0].weights))
    print("   biases: ", array_str(net.layers[0].biases))
    loss_values = main(net, inputs, expecteds, 1000, 0.2)
    print("final loss:", loss_values[-1])
    plt.plot(loss_values)
    plt.show()
