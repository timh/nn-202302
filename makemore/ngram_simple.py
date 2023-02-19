import torch
import torch.nn.functional as F
from typing import Callable

dictsize = 26 + 1

def make_data(numchar: int):
    names = open("names.txt").read().splitlines()

    res = list()

    num_names = len(names)
    num_chars = sum(len(name) for name in names)
    num_examples = num_chars + num_names

    inputs = torch.zeros((num_examples, numchar, dictsize))
    expected = torch.zeros((num_examples), dtype=torch.long)

    eidx = 0
    for nidx, name in enumerate(names):
        name_inputs = torch.zeros(len(name) + numchar, dictsize)
        for cidx, ch in enumerate(name):
            ch = torch.tensor(ord(ch) - ord('a'))
            name_inputs[numchar + cidx] = F.one_hot(ch, dictsize)
        for i in range(len(name) + 1):
            inputs[eidx] = name_inputs[i:i + numchar]
            if i < len(name):
                ch = torch.tensor(ord(name[i]) - ord('a'))
                expected[eidx] = ch
            else:
                expected[eidx] = torch.tensor(0)
            eidx += 1

    return inputs, expected

def samples(inputs: torch.Tensor, expected: torch.Tensor, batch_size: int):
    for offset in range(0, len(inputs), batch_size):
        step_inputs = inputs[offset:offset + batch_size]
        step_expected = expected[offset:offset + batch_size]

        yield step_inputs, step_expected

embeds_list = list()
l1in_list = list()
l1pre_list = list()
l1out_list = list()
l2out_list = list()
loss_list = list()

def main(epochs: int, lr_fun: Callable[[int], float],
         numchar = 3, numhidden = 30, numdims = 2, batch_size = 64,
         device = "cuda"):
    inputs_all, expected_all = make_data(numchar)

    emb_table = torch.randn((dictsize, numdims), device=device)
    l1_weights = torch.randn((numchar * numdims, numhidden), device=device) / ((numchar * numdims) ** 0.5)
    l1_biases = torch.randn((numhidden, ), device=device) * 0.01

    l2_weights = torch.randn((numhidden, numhidden), device=device) / (numhidden ** 0.5)
    l2_biases = torch.randn((numhidden, ), device=device) * 0.01

    emb_table.requires_grad_(True)
    l1_weights.requires_grad_(True)
    l1_biases.requires_grad_(True)
    l2_weights.requires_grad_(True)
    l2_biases.requires_grad_(True)

    print("-" * 20)
    print(f"{numchar=}")
    print(f"{numhidden=}")
    print(f"{numdims=}")

    print(f"{dictsize=}")
    print(f"{batch_size=}")
    
    learning_rates = [lr_fun(e) for e in range(epochs)]
    print()

    ntrain = int(len(inputs_all) * 0.9)
    inputs_train = inputs_all[:ntrain]
    expected_train = expected_all[:ntrain]

    inputs_val = inputs_all[ntrain:]
    expected_val = expected_all[ntrain:]

    embeds_list.clear()
    l1in_list.clear()
    l1out_list.clear()
    l2out_list.clear()
    loss_list.clear()
    def forward(binputs, bexpected):
        num_batch = len(binputs)
        # inputs = (B, numchar, dictsize)
        # expected = (B, 1)
        binputs = binputs.to(device)
        bexpected = bexpected.to(device)
        embeds = binputs @ emb_table                         # -> (B, numchar, numdim)
        l1in = embeds.view(num_batch, numchar*numdims)       # -> (B, numchar*numdim)
        l1pre = l1in @ l1_weights + l1_biases                # -> (B, numhidden)
        l1out = torch.tanh(l1pre)                            # -> (B, numhidden)
        l2out = l1out @ l2_weights + l2_biases               # -> (B, numhidden)
        loss = F.cross_entropy(l2out, bexpected)

        embeds_list.append(embeds)
        l1in_list.append(l1in)
        l1pre_list.append(l1pre)
        l1out_list.append(l1out)
        l2out_list.append(l2out)
        loss_list.append(loss)
        return loss

    for epoch in range(epochs):
        epoch_loss = 0.0
        count = 0
        lr = learning_rates[epoch]
        for bidx, (binputs, bexpected) in enumerate(samples(inputs_train, expected_train, batch_size)):
            loss = forward(binputs, bexpected)

            emb_table.grad = None
            l1_weights.grad = None
            l1_biases.grad = None
            loss.backward()

            emb_table.data -= lr * emb_table.grad
            l1_weights.data -= lr * l1_weights.grad
            l1_biases.data -= lr * l1_biases.grad

            epoch_loss += loss
            count += 1

        epoch_loss /= count
        with torch.no_grad():
            val_loss = forward(inputs_val, expected_val)
        print(f"{epoch+1}/{epochs}: train loss {epoch_loss:.4f}, val loss {val_loss:.4f} at lr={lr:.4f}")

if __name__ == "__main__":
    main(20, lambda epoch: 0.1)




