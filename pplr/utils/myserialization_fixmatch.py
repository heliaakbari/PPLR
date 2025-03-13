from __future__ import print_function, absolute_import
import json
import os.path as osp
import re
import shutil

from itertools import zip_longest


import torch
from torch.nn import Parameter

from .osutils import mkdir_if_missing


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        # checkpoint = torch.load(fpath)
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'), weights_only=False)

        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


import torch
from torch.nn import Parameter
from itertools import zip_longest
import re

def adjust_pretrain_keys(state_dict):
    """ Modify pretrain keys to match model keys """
    new_state_dict = {}

    for key, value in state_dict['model'].items():
        # Map conv1.weight -> module.base.0.weight
        if key == "conv1.weight":
            new_key = "module.base.0.weight"

        # Map bn1.* -> module.base.1.*
        elif key.startswith("bn1."):
            new_key = "module.base.1." + key[4:]  # Remove "bn1." and prepend "module.base.1."

        # Map classifier -> bnneck
        elif key == "classifier.weight":
            new_key = "module.bnneck.weight"
        elif key == "classifier.bias":
            new_key = "module.bnneck.bias"

        # Adjust layerX -> module.base.layer(X+3)
        else:
            match = re.match(r'layer(\d+)', key)
            if match:
                new_layer_num = int(match.group(1)) + 3
                new_key = f"module.base.{new_layer_num}" + key[len(match.group(0)):]
            else:
                new_key = key  # Keep unchanged

        new_state_dict[new_key] = value  # Assign value to new key
    if "module.bnneck.weight" in new_state_dict:
        new_state_dict.pop("module.bnneck.weight")
    if "module.bnneck.bias" in new_state_dict:
        new_state_dict.pop("module.bnneck.bias")
    return {'model': new_state_dict}  # Keep the same format


def copy_state_dict(state_dict, model, strip='module.'):
    """ Load pretrain weights while adjusting keys """
    state_dict = adjust_pretrain_keys(state_dict)  # Apply transformations

    tgt_state = model.state_dict()
    copied_names = set()

    model_keys = list(tgt_state.keys())
    pretrain_keys = list(state_dict['model'].keys())

    print("Displaying model and pretrain keys alternatively:\n")
    for m_key, p_key in zip_longest(model_keys, pretrain_keys, fillvalue="-- MISSING --"):
        print(f"Model Key: {m_key}  |  Pretrain Key: {p_key}")

    for name, param in state_dict['model'].items():
        if name not in tgt_state:
            print(f"{name} not in model state_dict")
            continue

        if isinstance(param, Parameter):
            param = param.data

        # Skip classifier if shape mismatch
        if "classifier" in name and param.size() != tgt_state[name].size():
            print(f"Skipping {name} due to shape mismatch: {param.size()} vs {tgt_state[name].size()}")
            continue

        tgt_state[name].copy_(param)

        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model
