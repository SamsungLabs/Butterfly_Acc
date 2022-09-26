from typing import Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import Linear, init
import time

from src.utils import low_rank_project
from torch_butterfly import Butterfly

class LowRank(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, rank: Union[int, float],
                 bias: bool=True, init='linear', weight_decay: bool = True,
                 device=None, dtype=None) -> None:
        """
        weight_decay: whether to mark the low-rank weights as _no_weight_decay.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if isinstance(rank, float):
            rank = int(rank * min(in_features, out_features))
        self.rank = rank
        self.lr_weight1 = Parameter(torch.empty((self.rank, in_features), **factory_kwargs))
        self.lr_weight2 = Parameter(torch.empty((out_features, self.rank), **factory_kwargs))
        if init not in ['linear', 'svd']:
            raise NotImplementedError(f'init {init} not supported')
        self.init = init

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        if not weight_decay:
            self.lr_weight1._no_weight_decay = True
            self.lr_weight2._no_weight_decay = True

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.init == 'linear':
                # Mimic torch.nn.Linear init
                init.kaiming_uniform_(self.lr_weight1, a=math.sqrt(5))
                init.kaiming_uniform_(self.lr_weight2, a=math.sqrt(5))
            elif self.init == 'svd':
                # Use spectral initialization as described in https://openreview.net/forum?id=KTlJT1nof6d
                full_weight = torch.nn.Linear(self.in_features, self.out_features, bias=False,
                                              device=self.lr_weight1.device,
                                              dtype=self.lr_weight1.dtype).weight
                self.set_weights_from_projection(full_weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.lr_weight1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    @property
    def saving(self):
        return ((self.in_features + self.out_features) * self.rank
                / (self.in_features * self.out_features))

    def set_weights_from_projection(self, weight):
        U, Vt = low_rank_project(weight, rank=self.rank)
        self.lr_weight1.copy_(Vt)
        self.lr_weight2.copy_(U)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(F.linear(input, self.lr_weight1), self.lr_weight2, self.bias)


class ButterflyLRLinear(nn.Module):
    def __init__(self, in_features, out_features,
                 bias=True, rank: Union[int, float] = 0.1,
                 gating=True, checkpointing=False):
        """If rank is float (e.g., 0.1), treat it as rank ratio.
        If rank is int (e.g., 32), treat it as rank.
        gating: whether to use sigmoid gating, otherwise we simply average the sparse and low-rank
        components.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.butterfly = Butterfly(in_features, out_features)
        self.low_rank = LowRank(in_features, out_features, rank=rank, bias=False)
        if gating:
            self.gate = nn.Linear(in_features=in_features, out_features=1)
        else:
            self.register_parameter('gate', None)
        self.checkpointing = checkpointing
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.bias is not None:
            fan_in = self.bias.shape[0]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def _multiply(self, x):
        butterfly_output = self.butterfly(x)
        low_rank_output = self.low_rank(x)
        g = torch.sigmoid(self.gate(x)) if self.gate is not None else 0.5
        # output = (1.0 - g) * sparse_output + g * low_rank_output
        return torch.lerp(butterfly_output, low_rank_output, g)

    def forward(self, x):
        if self.checkpointing:
            output = torch.utils.checkpoint.checkpoint(self._multiply, x)
        else:
            output = self._multiply(x)
        return (output + self.bias) if self.bias is not None else output


def unit_test(num_seq=512, hid_dim=768, ffn_dim = 3072, lr_ratio=0.1):
    # The speed of standard FFN
    device = "cuda"
    print ("If cuda avaiable:", torch.cuda.is_available())
    torch.no_grad()
    input = torch.randn(num_seq, hid_dim, device = device)
    runs = 1000
    std_time = 0.0
    std_linear1 = nn.Linear(hid_dim, ffn_dim).to(device)
    std_linear2 = nn.Linear(ffn_dim, hid_dim).to(device)
    for i in range(runs):
        begin_time = time.time()
        output = std_linear1(input)
        output = std_linear2(output)
        end_time = time.time()
        std_time += (end_time - begin_time)
    print ("Speed of dense linear:", std_time / runs)

    # The speed of Butterfly + Low Rank FFN
    bflylr_linear1 = ButterflyLRLinear(hid_dim, ffn_dim, rank=lr_ratio).to(device)
    bflylr_linear2 = ButterflyLRLinear(ffn_dim, hid_dim, rank=lr_ratio).to(device)
    bflylr_time = 0.0
    for i in range(runs):
        begin_time = time.time()
        output = bflylr_linear1(input)
        output = bflylr_linear2(output)
        end_time = time.time()
        bflylr_time += (end_time - begin_time)
    print ("Speed of sparse linear:", bflylr_time / runs, " with low rank rator:", lr_ratio)


if __name__ == "__main__":
    unit_test(lr_ratio=0.1)
    unit_test(lr_ratio=0.25)
    unit_test(lr_ratio=0.5)
