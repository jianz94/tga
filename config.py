# coding: utf-8
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='radoslaw',
                    help='dataset name.')
parser.add_argument('--num_nodes', type=int, default=151,
                    help='number of nodes in the network.')
parser.add_argument('--historical_len', type=int, default=2,
                    help='number of historical snapshots used for inference.')
parser.add_argument('--alpha', type=float, default=20,
                    help='alpha')
parser.add_argument('--beta', type=float, default=0.5,
                    help='beta')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size for training.')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of training epochs')
parser.add_argument('--gpu', type=str, default='-1',
                    help='GPU device.')
parser.add_argument('--target_snapshot', type=int, default=3,
                    help='target snapshot')
parser.add_argument('--write_results', type=bool, default=True,
                    help='whether to write the adversarial examples.')
parser.add_argument('--attack_mode', type=str, default=None)
parser.add_argument('--link_type', type=str, default='tp')
parser.add_argument('--disturbance_ratio', type=float, default=0)
args = parser.parse_args()

device = torch.device("cuda:" + args.gpu if (torch.cuda.is_available() and args.gpu != '-1') else "cpu")