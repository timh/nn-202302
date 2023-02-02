from typing import List
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from PIL import Image, ImageFont, ImageDraw

from layer import Layer, DenseLayer
from loss import Activation_Softmax_Loss_CategoricalCrossentropy as ASLCC
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

def main(net: Network, inputs: np.ndarray, expected: np.ndarray, steps: int, learning_rate: float) -> np.ndarray:
    loss_values = np.zeros((steps, ))

    last_print = datetime.datetime.now()
    for step in range(steps):
        now = datetime.datetime.now()
        rotated_inputs = inputs
        outputs = net.forward(rotated_inputs, expected)
        loss = net.loss_obj.loss
        loss_values[step] = loss

        outputs_str = array_str(net.loss_obj.output)
        exp_str = array_str(expected)
        if step == steps - 1:
            print(f"step {step}")
            print(f" outputs {outputs_str}\n"
                  f"expected {exp_str}\n"
                  f"    loss {loss:.4f}")

        dvalues = net.backward(expected)
        net.update_params(learning_rate)

        if now - last_print >= datetime.timedelta(seconds=1):
            last_print = now
            print(f"step {step}/{steps} | loss {loss:.4f}")

    return loss_values

if __name__ == "__main__":
    net = Network(num_inputs=2, neurons_hidden=10, layers_hidden=2, neurons_output=6, loss_obj=ASLCC())
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

    # train_inputs = np.random.default_rng().normal(0, 1.2, size=(1000, 2))
    train_inputs = (np.random.default_rng().random((2000, 2)) - 0.5) * 6.
    train_expected = expect_fun(train_inputs)

    loss_values = main(net, train_inputs, train_expected, 10000, 0.4)
    print("loss:", loss_values[-1])

    # test_inputs = np.random.default_rng().normal(0, 1.2, size=(5000, 2))
    test_inputs = (np.random.default_rng().random((2000, 2)) - 0.5) * 6.
    test_expected = expect_fun(test_inputs)

    test_outputs = net.forward(test_inputs, test_expected)
    test_loss = net.loss_obj.loss
    print("test_loss:", test_loss)

    for ridx in range(len(rings) + 1):
        ring_points = test_inputs[test_outputs[:, ridx] >= 0.5]
        plt.scatter(ring_points[:, 0], ring_points[:, 1], c=colors[ridx])

    # plt.plot(loss_values)
    plt.show()
