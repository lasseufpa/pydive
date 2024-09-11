import torch 
from torch import Tensor
from collections import OrderedDict
from pydive.types import Weights

import copy
import random


def subtract(input: Weights, other: Weights, use_states: bool=True) -> Weights:
    if use_states:
        difference = OrderedDict()
        if input.keys() != other.keys():
            raise ValueError("The state dicts have different keys.")

        for k in input.keys():
            difference[k] = torch.sub(input[k], other[k])

        return difference
    else:
        raise NotImplementedError


def adjust_by_coordinates(center: Weights, xdirection: Weights, ydirection: Weights, x: float, y: float, use_states: bool=True) -> Weights:
    if use_states:
        adjusted_state = OrderedDict()

        if center.keys() != xdirection.keys() != ydirection.keys():
            raise ValueError("The state dicts have different keys.")

        for k, v in center.items():
            adjusted_state[k] = v + x*xdirection[k] + y*ydirection[k]
        
        return adjusted_state
    else:
        raise NotImplementedError
    

def sqr_l2_norm(input: Weights, dtype: torch.dtype=torch.float, use_states=True) -> Tensor:

    if use_states:
        sqr_l2_norm = torch.tensor(0.0, dtype=dtype)
        
        for v in input.values():
            sqr_l2_norm += torch.sum(torch.inner(v.view(-1), v.view(-1)))

        return sqr_l2_norm
    
    else: 
        raise NotImplementedError


def l2_norm(input: Weights, dtype: torch.dtype=torch.float, use_states: bool=True) -> Tensor:

    if use_states:
        l2_norm = torch.tensor(0.0, dtype=dtype)

        for v in input.values():
            l2_norm += torch.sum(
                            torch.inner(v.view(-1), v.view(-1))
                            )
            
        l2_norm = torch.sqrt(l2_norm)

        return l2_norm
    
    else: 
        raise NotImplementedError


def vector_normalization(input: Weights, norm: float, use_states: bool=True) -> Weights:

    if use_states:
        normalized_state = OrderedDict()

        for k, v in input.items():
            normalized_state[k] = torch.div(v, norm)

        return normalized_state 
    
    else: 
        raise NotImplementedError


def orthogonal_projection(input: Weights, target: Weights, dtype: torch.dtype=torch.float, use_states: bool=True) -> Weights:

    if use_states:
        if input.keys() != target.keys():
            raise ValueError("The state dicts have different keys.")
        
        projected_state = OrderedDict()

        squared_input_norm = torch.tensor(0.0, dtype=dtype)
        input_inner_target = torch.tensor(0.0, dtype=dtype)
        
        for k in input.keys():
            squared_input_norm += torch.inner(input[k].view(-1), input[k].view(-1))
            input_inner_target += torch.inner(input[k].view(-1), target[k].view(-1))
        
        escalar_projection = torch.div(input_inner_target, squared_input_norm)

        print(squared_input_norm)
        print(input_inner_target)
        print(escalar_projection)

        for k, v in input.items():
            projected_state[k] = v * escalar_projection

        return projected_state
    
    else:
        raise NotImplementedError


def test_dot():
    random.seed(0)
    w = OrderedDict()
    w['layer1'] = torch.tensor([random.random()]*5)
    w['layer2'] = torch.tensor([random.random()]*5)

    base = OrderedDict()
    base['layer1'] = torch.tensor([random.random()]*5)
    base['layer2'] = torch.tensor([-random.random()]*5)
    result = OrderedDict()
    sum_ = 0.0
    for key in w.keys():
        result[key] = torch.dot(w[key].view(-1),base[key].view(-1))
        sum_ += result[key]


def adjust_optimizer(model, optimizer, deep=True):
    if deep:
        model = copy.deepcopy(model)
        optimizer = copy.deepcopy(optimizer)

    for group in optimizer.param_groups:
        group['params'] = list(model.parameters())
    
    if deep:
        return model, optimizer