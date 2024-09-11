import torch
from torch import Tensor

from typing import Union, Callable

from collections import OrderedDict


Dict = Union[dict, OrderedDict]
Number = Union[int, float] 
Weights = [Tensor, OrderedDict]
Objective = Union[torch.nn.Module, Callable]