import torch
import torch.nn.functional as F

numchar = 3
numhidden = 30
numdims = 2

dictsize = 26 + 1
batch_size = 64

def make_data():
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

def samples(inputs, expected):
    for offset in range(0, len(inputs), batch_size):
        step_inputs = inputs[offset:offset + batch_size]
        step_expected = expected[offset:offset + batch_size]

        yield step_inputs, step_expected

def main(inputs: torch.Tensor, expected: torch.Tensor, epochs: int, lr: float):

    emb_table = torch.randn((dictsize, numdims), requires_grad=True)
    l1_weights = torch.randn((numchar * numdims, numhidden), requires_grad=True)

    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        count = 0
        for bidx, (binputs, bexpected) in enumerate(samples(inputs, expected)):
            num_batch = len(binputs)

            # inputs = (B, numchar, dictsize)
            # expected = (B, 1)
            logits = binputs @ emb_table                      # -> (B, numchar, numdim)
            logits = logits.view(num_batch, numchar*numdims)  # -> (B, numchar*numdim)
            logits = logits @ l1_weights                      # -> (B, numhidden)
            loss = F.cross_entropy(logits, bexpected)

            emb_table.grad = None
            l1_weights.grad = None
            loss.backward()

            emb_table.data -= lr * emb_table.grad
            l1_weights.data -= lr * l1_weights.grad

            epoch_loss += loss
            count += 1
        epoch_loss /= count
        print(f"{epoch+1}/{epochs}: {epoch_loss:.4f}")

    
    print(f"{step} steps")

if __name__ == "__main__":
    inputs, expected = make_data()
    main(inputs, expected, 20, 0.1)




