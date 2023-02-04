from typing import List, Tuple
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime

import train

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_neurons = 64
    net = nn.Sequential(
        nn.Linear(1, n_neurons),
        nn.ReLU(),
        nn.Linear(n_neurons, n_neurons),
        nn.ReLU(),
        nn.Linear(n_neurons, 1),
    ).to(device)

    steps = 5000
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    input_fun = lambda start: [x for x in np.arange(start, start+math.pi*2, 0.1)]
    expect_fun = lambda vals: [math.sin(val) for val in vals]

    # train.
    train_vals = input_fun(0.)
    train_inputs = torch.tensor([[x] for x in train_vals], dtype=torch.float32, device=device)
    train_expected = torch.tensor([[x] for x in expect_fun(train_vals)], dtype=torch.float32, device=device)

    loss_values, train_outputs_all = train.train(net, loss_fn, optimizer, train_inputs, train_expected, steps)
    train_outputs = train_outputs_all[-1]

    # build test set.
    test_vals = input_fun(math.pi)
    test_inputs = torch.tensor([[x] for x in test_vals], dtype=torch.float32, device=device)
    test_expected = torch.tensor([[x] for x in expect_fun(test_vals)], dtype=torch.float32, device=device)

    test_outputs = net(test_inputs)
    test_loss = loss_fn(test_outputs, test_expected)

    # determine accuracy and plot
    acc_pred = torch.std(test_expected) / 250.
    acc = torch.mean(torch.absolute(test_outputs - test_expected) * 1.0 < acc_pred, dtype=torch.float32)

    print(f"test_loss: {test_loss:.4f}, acc: {acc:.4f}")

    train_outputs = train_outputs.detach().numpy()
    test_outputs = test_outputs.detach().numpy()
    plt.plot(test_outputs)
    plt.plot(train_outputs)
    plt.show()