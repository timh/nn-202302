from typing import List, Callable

import torch
import torch.nn as nn
import csv

class Input:
    cash_on_hand: torch.Tensor   # (1,)
    shares_on_hand: torch.Tensor # (1,)
    quotes: torch.Tensor         # (N,)

class Output:
    action: torch.Tensor  # (1,) - buy when >= 1.0
                          #      - sell when <= -1.0
    # the buy / sell are done at quotes[-1]

def make_net(num_quotes: int, num_hidden: int, hidden_size: int, device="cpu") -> nn.Module:
    mods: List[nn.Module] = list()

    mods.append(nn.Linear(num_quotes + 2, hidden_size))
    for _ in range(num_hidden):
        lin = nn.Linear(hidden_size, hidden_size)
        with torch.no_grad():
            lin.weight *= 10.0
        mods.append(lin)
        # mods.append(nn.BatchNorm1d(hidden_size))
        mods.append(nn.LeakyReLU())
    mods.append(nn.Linear(hidden_size, 1))

    net = nn.Sequential(*mods)
    return net.to(device)

# file
def read_quotes(filename: str) -> torch.Tensor:
    quoteslist: List[float] = list()

    # Date,Time,Open,High,Low,Close,Volume,OpenInt
    # 2015-11-24,15:35:00,6.99,6.99,6.92,6.95,5971,0
    # 2015-11-24,15:40:00,6.99,7.1,6.99,7.055,3000,0
    # 2015-11-24,15:45:00,7.02,7.0699,6.96,6.97,4900,0
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            quoteslist.append(float(row['Open']))
    
    return torch.tensor(quoteslist)

class TradingModule(nn.Module):
    starting_cash: torch.Tensor   # (1, )
    cash_on_hand: torch.Tensor    # (1, )
    shares_on_hand: torch.Tensor  # (1, )
    realnet: nn.Module            # the real net behind. it executes at most a single buy/sell.

    net_quotes_len: int           # how many quotes the net expects
    device: str

    def __init__(self, starting_cash: float, realnet: nn.Module, device="cpu"):
        super().__init__()
        self.realnet = realnet
        firstmod = list(self.realnet.modules())[1]
        self.net_quotes_len = firstmod.weight.shape[0]
        self.starting_cash = starting_cash
        self.device = device

    def forward(self, all_quotes: torch.Tensor) -> torch.Tensor:
        # quote_inputs are a full length.
        all_quotes_len = all_quotes.shape[-1]
        all_quotes = all_quotes.to(self.device)

        batch_len = all_quotes.shape[0]

        self.cash_on_hand = torch.ones((batch_len, ), device=self.device) * self.starting_cash
        self.cash_on_hand.requires_grad_(False)
        self.shares_on_hand = torch.zeros((batch_len, ), dtype=torch.int32, device=self.device)
        self.shares_on_hand.requires_grad_(False)

        timesteps = all_quotes_len - self.net_quotes_len
        inputs = torch.zeros((batch_len, self.net_quotes_len + 2), device=self.device)

        # print(f"forward:")
        # print(f"  {all_quotes.shape=}")
        # print(f"  {all_quotes_len=}")
        # print(f"  {timesteps=}")
        # print(f"  {inputs.shape=}")

        for ts in range(0, timesteps):
            # print(f"{ts:3}")
            # print(f"  cash {self.cash_on_hand}, shares {self.shares_on_hand}")
            # inputs[:, 0] = self.cash_on_hand
            # inputs[:, 1] = self.shares_on_hand
            # inputs[:, 2:] = all_quotes[:, ts:ts + self.net_quotes_len]
            
            output = self.realnet(inputs)
            one_action = output[:, 0]
            curprice = inputs[:, -1]

            # this will resolve to zero buys if one_action < 0
            # print(f"    {one_action=} {curprice=}")
            # print(f"    start   cash_on_hand {self.cash_on_hand}")
            # print(f"    start shares_on_hand {self.shares_on_hand}")
            # print(f"    {curprice=}")

            shares_to_buy = torch.maximum(torch.tensor(0), one_action)
            shares_to_buy = torch.minimum(self.cash_on_hand / curprice, shares_to_buy).int()
            # print(f"    {shares_to_buy=}")
            self.cash_on_hand -= curprice * shares_to_buy
            self.shares_on_hand += shares_to_buy

            shares_to_sell = torch.maximum(torch.tensor(0), -one_action)
            shares_to_sell = torch.minimum(self.shares_on_hand, shares_to_sell).int()
            # print(f"    {shares_to_sell=}")
            self.cash_on_hand += curprice * shares_to_sell
            self.shares_on_hand -= shares_to_sell

            # print(f"      end   cash_on_hand {self.cash_on_hand}")
            # print(f"      end shares_on_hand {self.shares_on_hand}")
        
        # return "holdings value" - (1,)
        return self.cash_on_hand + self.shares_on_hand * all_quotes[:, -1]

def loss_fn(outputs: torch.Tensor, _truth_unused: torch.Tensor=None):
    res = outputs.mean()
    res.requires_grad_(True)
    return -res
