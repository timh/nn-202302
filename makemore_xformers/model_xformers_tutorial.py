import math
from typing import Callable, Tuple, Dict
import re

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from model_utils import TextMapper, PositionalEncoding

class TransformerModel(nn.Module):
    # TODO: layernorm?
    def __init__(self, vocab_len: int, emb_len: int, nhead: int, 
                 nlayers: int, hidden_len: int, dropout: float, 
                 do_layernorm = True,
                 device="cpu"):
        super().__init__()
        
        if do_layernorm:
            self.lnorm1 = nn.LayerNorm(emb_len, device=device)
            self.lnorm2 = nn.LayerNorm(emb_len, device=device)
        else:
            self.lnorm1 = None
            self.lnorm2 = None

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(emb_len, dropout, device=device)
        encoder_layers = nn.TransformerEncoderLayer(emb_len, nhead, hidden_len, dropout, batch_first=True, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers, norm=self.lnorm1).to(device)
        self.encoder = nn.Embedding(vocab_len, emb_len, device=device)
        self.emb_len = emb_len
        self.decoder = nn.Linear(emb_len, vocab_len, device=device)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs: Tensor, input_mask: Tensor = None) -> Tensor:
        """
        Args:
            inputs: Tensor, shape [batch_size, seq_len]
            input_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, vocab_len]
        """
        seq_len = inputs.shape[-1]
        if input_mask is None:
            input_mask = generate_square_subsequent_mask(seq_len, device=inputs.device)
        # print(f"{input_mask=}")
        inputs = self.encoder(inputs) * math.sqrt(self.emb_len)
        inputs = self.pos_encoder(inputs)

        output = self.transformer_encoder(inputs, input_mask)
        if self.lnorm2 is not None:
            output = self.lnorm2(output)
        output = self.decoder(output)

        return output
    

def generate_square_subsequent_mask(sz: int, device="cpu") -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    # return torch.triu(torch.ones(sz, sz) * True, diagonal=1).to(device=device, dtype=torch.bool)
    return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)

def loss_fn(seq_len: int, vocab_len: int) -> Callable[[Tensor, Tensor], Tensor]:
    def ce(outputs: Tensor, truth: Tensor) -> Tensor:
        batch_size = outputs.shape[0]
        outflat = outputs.view(batch_size * seq_len, vocab_len)
        truthflat = truth.view(batch_size * seq_len)
        return F.cross_entropy(outflat, truthflat)

    return ce

RE_AFTER_BASENAME = re.compile(r"[\w\d-]+-(.*)\.torch")
def _parse_model_filename(model_filename: str) -> Dict[str, str]:
    if model_filename.startswith("runs/"):
        model_filename = model_filename[5:]
    match = RE_AFTER_BASENAME.match(model_filename)
    fields_str = match.group(1)

    fields_list = fields_str.split(", ")
    fields: Dict[str, str] = dict()
    fields["filename"] = model_filename
    for field_str in fields_list:
        key, value = field_str.split(" ")
        # if key in FIELDS_LONG:
        #     key = FIELDS_LONG[key]
        fields[key] = value
    
    return fields

def load_model_and_textmap(model_filename: str, text_filename: str) -> Tuple[TransformerModel, TextMapper]:
    model_fields = _parse_model_filename(model_filename)
    model: TransformerModel = torch.load(model_filename)

    seq_len = int(model_fields["seqlen"])
    wordmaxlen = int(model_fields["wordlen"])
    textmap = TextMapper(seq_len=seq_len, filename=text_filename, wordmaxlen=wordmaxlen, device="cuda", dtype=torch.long)

    return model, textmap

FIELDS_SHORT = {
    "seq_len": "seqlen",
    "wordmaxlen": "wordlen",
    "nhead": "nhead",
    "nlayers": "nlayers",
    "hidden_len": "hidlen",
    "emb_len": "emblen",
    "do_layernorm": "norm",
    "optim_type": "optim",
    "start_lr": "startlr",
    "end_lr": "etartlr",
    "compile": "compile",
    "minibatch_count": "minicnt",
    "dropout": "dropout",
    "batch_size": "batch",
    "total_epochs": "epochs",
    "vocab_len": "vocablen",
}
FIELDS_LONG = {val: key for key, val in FIELDS_SHORT.items()}
