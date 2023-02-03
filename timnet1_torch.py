from typing import List
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime

def array_str(array: torch.Tensor) -> str:
    if len(array.shape) > 1:
        child_strs = [array_str(child) for child in array]
        child_strs = ", ".join(child_strs)
    else:
        child_strs = [format(v, ".4f") for v in array]
        child_strs = ", ".join(child_strs)
    return f"[{child_strs}]"

def train(network: nn.Module, inputs: torch.Tensor, expected: torch.Tensor, steps: int, learning_rate: float) -> torch.Tensor:
    loss_values = torch.zeros((steps, ))

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    first_print = last_print = datetime.datetime.now()
    last_step = 0
    for step in range(steps):
        now = datetime.datetime.now()
        outputs = network(inputs)

        loss = loss_fn(outputs, expected)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        delta_last = now - last_print
        if delta_last >= datetime.timedelta(seconds=1):
            delta_first = now - first_print
            persec_first = step / delta_first.total_seconds()
            persec_last = (step - last_step) / delta_last.total_seconds()
            last_print = now
            last_step = step
            print(f"step {step}/{steps} | loss {loss:.4f} | {persec_last:.4f}, {persec_first:.4f} overall")

    return loss_values

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = nn.Sequential(
        nn.Linear(2, 60),
        nn.ReLU(),
        nn.Linear(60, 60),
        nn.ReLU(),
        nn.Linear(60, 60),
        nn.ReLU(),
        nn.Linear(60, 6),
    ).to(device)
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
        res = torch.zeros((len(inputs), len(rings) + 1))
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

    train_inputs = (np.random.default_rng().random((10000, 2)) - 0.5) * 8.
    train_inputs = torch.from_numpy(train_inputs).to(device, torch.float32)
    train_expected = expect_fun(train_inputs).to(device)

    print(f"train_inputs {train_inputs}")
    print(f"train_expected {train_expected}")

    loss_values = train(net, train_inputs, train_expected, 10000, 0.1)

    # test_inputs = np.random.default_rng().normal(0, 1.2, size=(5000, 2))
    test_inputs = (np.random.default_rng().random((2000, 2)) - 0.5) * 8.
    test_inputs = torch.from_numpy(test_inputs).to(device, torch.float32)
    test_expected = expect_fun(test_inputs).to(device)

    test_outputs = net(test_inputs)
    test_loss = nn.CrossEntropyLoss()(test_outputs, test_expected)
    print(f"test_loss: {test_loss:.4f}")

    if device != "cpu":
        test_inputs = test_inputs.to("cpu")
        test_outputs = test_outputs.to("cpu")

    for ridx in range(len(rings) + 1):
        ring_points = test_inputs[test_outputs[:, ridx] >= 0.5]
        plt.scatter(ring_points[:, 0], ring_points[:, 1], c=colors[ridx])

    # plt.plot(loss_values)
    # plt.show()
    plt.savefig("timnet1_torch.png")
