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

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

MAX_MODIFIED_LINKS = 10
num_modified_links = []
num_succeed = 0

# load data and target links to be attacked
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
net = net.to(device)
net.load_state_dict(torch.load('models/{}_len_{}_params.pkl'.format(args.dataset, args.historical_len)))

criterion_masked = MaskedLoss()

pbar = tqdm(enumerate(target_links), total=len(target_links))
for k, (s, t) in pbar:
    # iter-No.
    attack_history = {}
    original_prob, min_prob = 0, 0
    adv_examples, probs, grads = {}, {}, {}
    pbar.set_description('{:.4f}/{:.4f}'.format(original_prob, min_prob))
    for his_s, _, true_s, _, _ in generate_batch(graphs,
                                                 [(s, 0, 0, 0)],
                                                 args.target_snapshot - args.historical_len,
                                                 1, is_train=False, is_cuda=True):
        num_iter = 0
        is_success = False

        inputs = Variable(his_s, requires_grad=True)
        # prob: (num_samples, num_nodes)
        prob, _ = net(inputs)
        original_prob = prob[0, t].data
        masked_loss = criterion_masked(true_s, prob, t)
        grad = torch.autograd.grad(masked_loss, inputs, retain_graph=False)[0].cpu().detach().data

        # iteration #0
        adv_examples[num_iter] = [his_s.cpu().data]
        probs[num_iter] = [prob[:, t]]
        if args.disturbance_ratio > 0:
            grads[num_iter] = [gradient_disturb(grad.squeeze(), args.disturbance_ratio)]
        else:
            grads[num_iter] = [grad.squeeze()]
        attack_history[num_iter] = [[]]

        # iteration #1 ~ #MAX_MODIFIED_LINKS-1
        while not is_success and num_iter < MAX_MODIFIED_LINKS:
            num_iter += 1
            grads[num_iter] = []
            adv_examples[num_iter] = []
            attack_history[num_iter] = []
            for i in range(len(adv_examples[num_iter - 1])):
                for ts in range(args.historical_len):
                    _adv_example, ah = one_step_attack(grads[num_iter - 1][i],
                                                       adv_examples[num_iter - 1][i],
                                                       attack_history[num_iter - 1][i], ts)
                    adv_examples[num_iter].append(_adv_example)
                    attack_history[num_iter].append(ah)

            if len(adv_examples[num_iter]) < args.batch_size:
                _inputs = torch.cat(adv_examples[num_iter], dim=1)
                _inputs = Variable(_inputs, requires_grad=True).to(device)
                prob, _ = net(_inputs)
                probs[num_iter] = prob[:, t].cpu().detach().numpy().tolist()
                masked_loss = criterion_masked(true_s, prob, t)
                grad = torch.autograd.grad(masked_loss, _inputs, retain_graph=False)[0].squeeze(-2).cpu().detach().data
            else:
                _tmp_grad = []
                for _e in range(len(adv_examples[num_iter]) // args.batch_size):
                    _inputs = torch.cat(adv_examples[num_iter][_e*args.batch_size:(_e+1)*args.batch_size], dim=1)
                    _inputs = Variable(_inputs, requires_grad=True).to(device)
                    prob, _ = net(_inputs)
                    probs[num_iter] = prob[:, t].cpu().detach().numpy().tolist()
                    masked_loss = criterion_masked(true_s, prob, t)
                    grad = torch.autograd.grad(masked_loss, _inputs, retain_graph=False)[0].squeeze(-2).cpu().detach().data
                    _tmp_grad.append(grad)
                if len(adv_examples[num_iter]) // args.batch_size != 0:
                    _inputs = torch.cat(adv_examples[num_iter][-(len(adv_examples[num_iter]) % args.batch_size):], dim=1)
                    _inputs = Variable(_inputs, requires_grad=True).to(device)
                    prob, _ = net(_inputs)
                    probs[num_iter] = prob[:, t].cpu().detach().numpy().tolist()
                    masked_loss = criterion_masked(true_s, prob, t)
                    grad = torch.autograd.grad(masked_loss, _inputs, retain_graph=False)[0].squeeze(-2).cpu().detach().data
                    _tmp_grad.append(grad)
                grad = torch.cat(_tmp_grad, dim=1)
            for i in range(grad.shape[1]):
                if args.disturbance_ratio > 0:
                    grads[num_iter].append(gradient_disturb(grad[:, i, :], args.disturbance_ratio))
                else:
                    grads[num_iter].append(grad[:, i, :])

            min_prob = min(probs[num_iter])
            if min_prob <= 0.5:
                if not is_success:
                    num_succeed += 1
                    num_modified_links.append(num_iter)
                is_success = True
            if num_iter >= 2:
                del adv_examples[num_iter - 2]
                del grads[num_iter - 2]
        if not is_success:
            num_modified_links.append(MAX_MODIFIED_LINKS)
print('ASR: {:.4f} AML: {:.4f} '.format(num_succeed / num_target_links, np.mean(num_modified_links)))
with open('tra_res.csv', 'a') as f:
    f.write('{}-{}, ASR: {:.4f}, AML: {:.4f}\n'.format(args.dataset, args.disturbance_ratio,
                                                       num_succeed / num_target_links, np.mean(num_modified_links)))
