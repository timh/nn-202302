from typing import List
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
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

    last_print = datetime.datetime.now()
    loss_cc = ASLCC()
    for step in range(steps):
        now = datetime.datetime.now()
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

        if now - last_print >= datetime.timedelta(seconds=1):
            last_print = now
            print(f"step {step}/{steps} | loss {loss:.4f}")

        # html_util.draw_step(net, out)
        # print()
    
    # print("</body>", file=out)
    # print("</html>", file=out)

    return loss_values

if __name__ == "__main__":
    net = Network(num_inputs=2, neurons_hidden=10, layers_hidden=2, neurons_output=6)
    width, height = 16, 16

    rings = [
        [-2,  1],
        [-1, -1],
        [ 0,  1],
        [ 1, -1],
        [ 2,  1]
    ]
    colors = [
        "gray",
        "blue",
        "yellow",
        "black",
        "green",
        "red"
    ]

    def expect_fun(inputs):
        inner_radius = 0.5
        outer_radius = 0.7
        res = np.zeros((len(inputs), len(rings) + 1))
        for iidx, (x, y) in enumerate(inputs):
            found_ring = False
            for ridx, (ringx, ringy) in enumerate(rings):
                testx, testy = x - ringx, y - ringy
                testrad = testx**2 + testy**2
                if testrad >= inner_radius and testrad < outer_radius:
                    res[iidx][ridx + 1] = 1.
                    found_ring = True
                    break
            if not found_ring:
                res[iidx][0] = 1.

        return res

    train_inputs = np.random.default_rng().normal(0, 1.5, size=(1000, 2))
    train_expected = expect_fun(train_inputs)
    # train_inputs = np.reshape(train_inputs, (-1, 1))

    loss_values = main(net, train_inputs, train_expected, 10000, 0.5)
    print("loss:", loss_values[-1])

    test_inputs = np.random.default_rng().normal(0, 1.5, size=(5000, 2))
    test_expected = expect_fun(test_inputs)
    # test_inputs = np.reshape(test_inputs, (-1, 1))

    test_outputs = net.forward(test_inputs)
    aslcc = ASLCC()
    test_loss = aslcc.forward(test_outputs, test_expected)
    test_outputs = aslcc.output
    # print("test_inputs:", array_str(test_inputs))
    # print("test_expected:", array_str(test_expected))
    # print("test_outputs:", array_str(test_outputs))
    print("test_loss:", test_loss)


    # inside = test_inputs[test_outputs[:, 0] >= 0.6]
    # outside = test_inputs[test_outputs[:, 0] < 0.6]
    # # plt.plot(inside, color="green")
    # # plt.plot(outside, color="red")
    # plt.scatter(outside[:, 0], outside[:, 1], c="red")
    # plt.scatter(inside[:, 0], inside[:, 1], c="green")

    for ridx in range(len(rings) + 1):
        ring_points = test_inputs[test_outputs[:, ridx] >= 0.5]
        plt.scatter(ring_points[:, 0], ring_points[:, 1], c=colors[ridx])

    # plt.plot(loss_values)
    plt.show()
