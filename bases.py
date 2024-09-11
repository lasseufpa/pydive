import torch
from torch import Tensor
import copy

from collections import OrderedDict
from typing import Union

from pydive.spatial import Plane
from pydive.utils import linalg

from pydive.types import Number, Dict, Weights, Objective


class Izmailov(Plane):
    def __init__(self, w1=None, w2=None, w3=None, dtype: torch.dtype=torch.float, use_states: bool=True, device: str='cpu'):

        if w1 is None:
            raise ValueError("W1 can't be None")
        if w2 is None:
            raise ValueError("W2 can't be None")
        if w3 is None:
            raise ValueError("W3 can't be None")
        
        # vector_dict{1: w1, 2: w2, 3: w3}

        # for k in vector_dict.keys():
        #     vector_dict[k] = remove_n_averaged(vector_dict[k])
        #     vector_dict[k] = remove_key_prefix(vector_dict[k], 'module.')
        
        # w1 = vector_dict[1]
        # w2 = vector_dict[2]
        # w3 = vector_dict[3]

        # del vector_dict

        # initialization
        self.device = device
        
        self.w1 = copy.deepcopy(w1)
        self.w2 = copy.deepcopy(w2) 
        self.w3 = copy.deepcopy(w3)
        self.dtype = dtype
        self.use_states = use_states
        # create u and v base vectors
        self.u = linalg.subtract(self.w2, self.w1)
        self.v = linalg.subtract(self.w3, self.w1)
        # project u onto v
        v_onto_u = linalg.orthogonal_projection(self.u, self.v, dtype=dtype)
        v_minus_projected_u = linalg.subtract(self.v, v_onto_u)
        # normalization
        self.u_norm = linalg.l2_norm(self.u, dtype=dtype)
        self.v_norm = linalg.l2_norm(v_minus_projected_u, dtype=dtype)
        self.center = copy.deepcopy(w1)
        self.xdirection = linalg.vector_normalization(self.u, self.u_norm)
        self.ydirection = linalg.vector_normalization(v_minus_projected_u, self.v_norm)

    def adjust_weights(self, x: Number=0.0, y: Number=0.0, dtype: torch.dtype=None, use_states=True):
        
        dtype = dtype if dtype else self.dtype

        if use_states:
            return linalg.adjust_by_coordinates(self.center, self.xdirection, self.ydirection, x, y)
        else:
            raise NotImplementedError
            
    def weights_projection(self, weights: Weights, dtype: torch.dtype=None, use_states=True):

        dtype = dtype if dtype else self.dtype

        if use_states:
            x = torch.tensor(0.0, dtype=dtype, device=self.device)
            y = torch.tensor(0.0, dtype=dtype, device=self.device)

            w = linalg.subtract(weights, self.w1)

            for k in w.keys():
                x += torch.inner(w[k].view(-1), self.xdirection[k].view(-1))
                y += torch.inner(w[k].view(-1), self.ydirection[k].view(-1))

            return torch.tensor((x, y), dtype=dtype)
        
        else:
            raise NotImplementedError
    