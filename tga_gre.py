# coding: utf-8
import os
import torch
import pickle as pkl
from torch.autograd import Variable
from model import ddne, MaskedLoss
from config import args, device
from attacker import *
from utils import *
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

MAX_MODIFIED_LINKS = 10
# MAX_MODIFIED_LINKS = min([200, round((args.num_nodes-1)*0.01)])
# if MAX_MODIFIED_LINKS < 10:
#     MAX_MODIFIED_LINKS = 10
num_modified_links = []
num_succeed = 0

graphs, _ = load_data(args.dataset, args.historical_len)
with open('results/{}/{}/{}/{}_target_links.list'.format(args.dataset, args.link_type,
                                                         args.historical_len,
                                                         args.target_snapshot), 'rb') as f:
    target_links = pkl.load(f)
num_target_links = len(target_links)
with open('models/{}.model'.format(args.dataset), 'r') as f:
    model_definition = [int(x) for x in f.readline().split(',')]
args.num_nodes = model_definition[-1]
net = ddne(model_definition[0], model_definition[1:])
net.load_state_dict(torch.load('models/{}_len_{}_params.pkl'.format(args.dataset, args.historical_len, map_location={'cuda:3': 'cuda:0'})))
net.eval()

model = ddne(model_definition[0], model_definition[1:])
model.load_state_dict(torch.load('models/{}_len_{}_params.pkl'.format(args.dataset, args.historical_len)))
model = model.to(device)
model.eval()

criterion_masked = MaskedLoss()
pbar = tqdm(target_links)
for (s, t) in pbar:
    pbar.set_description('{}'.format(MAX_MODIFIED_LINKS))
    # just focus on the source node
    attack_history = list()
    original_prob, min_prob = 0, 0
    for his_s, _, true_s, _, _ in generate_batch(graphs,
                                                 [(s, 0, 0, 0)],
                                                 args.target_snapshot - args.historical_len,
                                                 1, is_train=False, is_cuda=False):
        num_iter = 0
        is_success = False

        inputs = Variable(his_s, requires_grad=True).to('cpu')

        # prob: (num_samples, num_nodes)
        prob, _ = net(inputs)
        original_prob = prob[0, t].data
        masked_loss = criterion_masked(true_s, prob, t)
        grad = torch.autograd.grad(masked_loss, inputs, retain_graph=False)[0].data

        # iteration #0
        best_adv_example = copy.deepcopy(his_s)
        min_prob = prob[:, t]
        if args.disturbance_ratio > 0:
            best_grad = gradient_disturb(grad.squeeze(), args.disturbance_ratio)
        else:
            best_grad = grad.squeeze()

        # iteration #1 ~ #MAX_MODIFIED_LINKS-1
        while not is_success and num_iter < MAX_MODIFIED_LINKS:
            num_iter += 1
            ah, input_list = [], []
            for ts in range(args.historical_len):
                _adv_examples, _h = one_step_attack(best_grad, best_adv_example,
                                                    attack_history, ts,
                                                    mode=args.attack_mode)
                input_list.append(_adv_examples)
                ah.append(_h)
            _inputs = torch.cat(input_list, 1).to(device)
            probs, _ = model(_inputs)
            min_idx = torch.argmin(probs[:, t]).data

            best_adv_example = copy.deepcopy(input_list[min_idx])
            _inputs = Variable(best_adv_example, requires_grad=True)
            prob, _ = net(_inputs)
            masked_loss = criterion_masked(true_s, prob, t)
            grad = torch.autograd.grad(masked_loss, _inputs, retain_graph=False)[0].data
            if args.disturbance_ratio > 0:
                best_grad = gradient_disturb(grad.squeeze(), args.disturbance_ratio)
            else:
                best_grad = grad.squeeze()

            min_prob = probs[min_idx, t].data
            attack_history += ah[min_idx]
            if min_prob <= 0.5:
                if not is_success:
                    num_succeed += 1
                    num_modified_links.append(num_iter)
                is_success = True
        if not is_success:
            # print('{:.4f}/{:.4f}'.format(original_prob, min_prob))
            num_modified_links.append(MAX_MODIFIED_LINKS)

print('ASR: {:.4f}'.format(num_succeed/num_target_links),
      'AML: {:.4f} '.format(np.mean(num_modified_links)))

with open('res.csv', 'a') as f:
    f.write('{}-{}, ASR: {:.4f}, AML: {:.4f}\n'.format(args.dataset, args.disturbance_ratio,
                                                       num_succeed/num_target_links, np.mean(num_modified_links)))
