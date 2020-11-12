# coding: utf-8
import torch
import pickle as pkl
import numpy as np
import random
import copy
from torch.autograd import Variable
from config import args, device


def load_data(network, h_len):

    with open('data/{}.network'.format(network), 'rb') as f:
        graphs = pkl.load(f)

    with open('data/{}_len_{}.train'.format(network, h_len), 'rb') as f:
        node_pairs = pkl.load(f)

    return graphs, node_pairs


def generate_batch(graphs, node_pairs, start, batch_size, is_train=True, is_cuda=False):
    graph = graphs[start + args.historical_len].toarray()
    if is_train:
        random.shuffle(node_pairs)
    num_batches = len(node_pairs) // batch_size
    for i in range(num_batches):
        input_s, input_t = [[] for _ in range(args.historical_len)], [[] for _ in range(args.historical_len)]
        true_s, true_t, weight = [], [], []
        for s, t, _, w in node_pairs[i*batch_size: (i+1)*batch_size]:
            true_s.append(graph[s])
            true_t.append(graph[t])
            weight.append(w)
            for k in range(args.historical_len):
                # historical_graph = graphs[start + k].toarray()
                input_s[k].append(graphs[start + k][s].toarray())
                input_t[k].append(graphs[start + k][t].toarray())
        input_s = torch.FloatTensor(np.array([np.expand_dims(np.vstack(input_s[i]), axis=1)
                                              for i in range(args.historical_len)]))
        input_t = torch.FloatTensor(np.array([np.expand_dims(np.vstack(input_t[i]), axis=1)
                                              for i in range(args.historical_len)]))
        true_s = torch.FloatTensor(np.array(true_s)).squeeze(dim=1)
        true_t = torch.FloatTensor(np.array(true_t)).squeeze(dim=1)
        weight = torch.FloatTensor(np.reshape(weight, (-1, 1)))
        if is_cuda:
            yield input_s.to(device), input_t.to(device), true_s.to(device), true_t.to(device), weight.to(device)
        else:
            yield input_s, input_t, true_s, true_t, weight


def gradient_disturb(gradients: torch.FloatTensor, ratio):
    max_gradient_norm = torch.max(torch.abs(gradients)).item()
    num_disturbance = int(args.historical_len * args.num_nodes * 0.3)
    disturbance_value = ratio*max_gradient_norm * np.ones((num_disturbance, ))
    disturbance_value = disturbance_value.astype(np.float)
    disturbance_indices = random.sample(range(args.num_nodes), num_disturbance)
    disturbance_value = torch.zeros_like(gradients).index_put_((torch.LongTensor([random.sample(range(args.historical_len), 1)
                                                                for _ in range(num_disturbance)]),
                                                                torch.LongTensor(disturbance_indices)),
                                                               torch.FloatTensor(disturbance_value))
    return gradients + disturbance_value


def weights_init(m: torch.nn.Module):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.GRU):
        torch.nn.init.xavier_uniform_(m.all_weights[0][0])
        torch.nn.init.xavier_uniform_(m.all_weights[0][1])
        torch.nn.init.constant_(m.all_weights[0][2], 0)
        torch.nn.init.constant_(m.all_weights[0][3], 0)


def adv_test(net, inputs):
    grads, probs = [], []
    num_adv_examples = int(inputs.shape[1])
    for k in range(num_adv_examples // args.batch_size + 1):
        _inputs = Variable(inputs[:, k*args.batch_size:(k+1)*args.batch_size], requires_grad=True)
        prob, _ = net(_inputs)
        probs.append(prob)
    return probs
