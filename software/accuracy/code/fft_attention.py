import functools
import math
import copy
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from functools import partial

from einops import repeat

from transformers.file_utils import is_scipy_available

if is_scipy_available():
    from scipy import linalg


class FNetBasicFourierTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self._init_fourier_transform()

    def _init_fourier_transform(self):
        self.fourier_transform = partial(torch.fft.fftn, dim=(1, 2))

    def forward(self, hidden_states):

        # NOTE: We do not use torch.vmap as it is not integrated into PyTorch stable versions.
        # Interested users can modify the code to use vmap from the nightly versions, getting the vmap from here:
        # https://pytorch.org/docs/master/generated/torch.vmap.html. Note that fourier transform methods will need
        # change accordingly.

        outputs = self.fourier_transform(hidden_states).real
        return (outputs,)