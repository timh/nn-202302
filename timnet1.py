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
    loss_values = np.zeros((steps, ))

    # out = open(f"steps.html", "w")
    # print("<html>", file=out)
    # print("<head>", file=out)
    # print("<link rel=\"stylesheet\" href=\"net.css\"></link>", file=out)
    # print("</head>", file=out)
    # print("<body>", file=out)

    loss_cc = ASLCC()
    for step in range(steps):
        rotated_inputs = inputs
        outputs = net.forward(rotated_inputs)
        loss = loss_cc.forward(outputs, expected)
        loss_values[step] = loss

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
        
        net.update(learning_rate)
        # html_util.draw_step(net, out)
        # print()
    
    # print("</body>", file=out)
    # print("</html>", file=out)

    return loss_values

if __name__ == "__main__":
    net = Network(num_inputs=2, neurons_hidden=20, layers_hidden=4, neurons_output=2)
    # input_fun = lambda vals: [x if x % 2 == 0 else -x for x in vals]
    expect_fun = lambda inputs: np.array([[1, 0] if (x**2 + y**2) <= 1.0 else [0, 1] for x, y in inputs])

    train_inputs = np.random.default_rng().normal(0, 1.0, size=(30, 2))
    train_expected = expect_fun(train_inputs)
    # train_inputs = np.reshape(train_inputs, (-1, 1))

    loss_values = main(net, train_inputs, train_expected, 1000, 0.2)
    print("loss:", loss_values[-1])

    test_inputs = np.random.default_rng().normal(0, 1.0, size=(5, 2))
    test_expected = expect_fun(test_inputs)
    # test_inputs = np.reshape(test_inputs, (-1, 1))

    test_outputs = net.forward(test_inputs)
    aslcc = ASLCC()
    test_loss = aslcc.forward(test_outputs, test_expected)
    test_outputs = aslcc.output
    print("test_inputs:", array_str(test_inputs))
    print("test_expected:", array_str(test_expected))
    print("test_outputs:", array_str(test_outputs))
    print("test_loss:", test_loss)


    plt.plot(loss_values)
    plt.show()
