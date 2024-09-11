import torch

from collections import OrderedDict

from pydive.types import Dict

import copy


def remove_key_prefix(state_dict: Dict, prefix: str) -> OrderedDict:
    """Removes the starting 'prefix' from the key string of the state_dict"""
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith(prefix):
            idx = len(prefix)
            new_state_dict[k[idx:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def remove_n_averaged(state_dict: OrderedDict) -> OrderedDict:
    """Returns a new state without the element 'n_averaged', if it exists"""

    new_state = OrderedDict()
    
    for k, v in state_dict.items():
        if 'n_averaged' in k:
            continue    
        else:
            new_state[k] = v

    return new_state


def split_state_by_key(state_dict: Dict, key: str=None) -> OrderedDict:
    state_key, state_left = OrderedDict(), OrderedDict()

    if key == None:
        raise ValueError('No key to split state')
    
    else:
        for k in state_dict.keys():
            if key in k:
                state_key[k] = copy.deepcopy(state_dict[k])
            else:
                state_left[k] = copy.deepcopy(state_dict[k])

    return state_key, state_left


def to_type(state: Dict, dtype: torch.dtype=torch.float64) -> OrderedDict:
    """Pass the tensors of a state to a specific pytorch type"""
    
    new_state = copy.deepcopy(state)

    for k, v in state.keys():
        new_state[k] = v.to(dtype)
    
    return new_state