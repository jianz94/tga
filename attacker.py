# coding: utf-8
import numpy as np
import random
import copy
import torch
from config import args
from collections import OrderedDict


def one_step_attack(gradients, inputs, attack_history, ts, mode=None):
    # gradients: (historical_len, num_nodes)
    # inputs: (historical_len, 1, ,1, num_nodes)
    # max_grad = torch.FloatTensor([torch.max(torch.abs(gradients[ts]))])
    # min_grad = torch.FloatTensor([torch.min(torch.abs(gradients[ts]))])
    # rescaled_grad = max_grad - gradients / max_grad - min_grad
    _, sorted_index = torch.sort(torch.abs(gradients[ts]), descending=True)
    sorted_index = sorted_index.squeeze()
    _inputs = copy.deepcopy(inputs)
    history = []
    for i in range(args.num_nodes):
        value = is_to_modify(gradients[ts][sorted_index[i]], inputs[ts, 0, 0, sorted_index[i]])
        if not mode and value != -1:
            if (ts, sorted_index[i], value) not in attack_history and len(history) < 5:
                _inputs = _inputs.index_put_((torch.LongTensor([ts]), torch.LongTensor([0]),
                                              torch.LongTensor([0]), torch.LongTensor([sorted_index[i]])),
                                             torch.FloatTensor([value]))
                history.append(ts, sorted_index[i], value)
                break
        elif mode == 'add' and (ts, sorted_index[i]) not in attack_history  and len(history) < 5:
            if inputs[ts, 0, 0, sorted_index[i]] == 0:
                _inputs = _inputs.index_put_((torch.LongTensor([ts]), torch.LongTensor([0]),
                                              torch.LongTensor([0]), torch.LongTensor([sorted_index[i]])),
                                             torch.FloatTensor([1]))
                history.append(ts, sorted_index[i], 1)
                break
    return _inputs, history


def random_attack(inputs):
    _inputs = copy.deepcopy(inputs)
    node = random.choice(range(args.num_nodes))
    ts = random.choice(range(args.historical_len))
    value = 1 if inputs[ts, :, :, node] == 0 else 0
    return _inputs.index_put((torch.LongTensor([ts]), torch.LongTensor([0]),
                              torch.LongTensor([0]), torch.LongTensor([node])),
                             torch.FloatTensor([value]))


def is_to_modify(g, link):
    if g > 0 and link == 1:
        modify = 0
    elif g <= 0 and link == 0:
        modify = 1
    else:
        modify = -1

    return modify
