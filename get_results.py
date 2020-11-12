# coding: utf-8
import os
import shutil
import sys
import time

import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse import csr_matrix
from model import ddne
from utils import *
from config import args, device
import networkx as nx
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

graphs, _ = load_data(args.dataset, args.historical_len)
with open('models/{}.model'.format(args.dataset), 'r') as f:
    model_definition = [int(x) for x in f.readline().split(',')]
args.num_nodes = model_definition[-1]
net = ddne(model_definition[0], model_definition[1:]).to(device)
net.load_state_dict(torch.load('models/{}_len_{}_params.pkl'.format(args.dataset, args.historical_len)))
net.eval()
probs, labels = [], []
for k in tqdm(range(args.num_nodes // args.batch_size)):
    for his_s, _, true_s, _, _ in generate_batch(graphs,
                                                 [(k*args.batch_size + i, 0, 0, 0) for i in range(args.batch_size)],
                                                 args.target_snapshot-args.historical_len,
                                                 args.batch_size, is_train=False, is_cuda=True):
        with torch.no_grad():
            pred, _ = net(his_s)
            probs.append(pred.cpu().numpy())
if args.num_nodes % args.batch_size != 0:
    tmp = args.num_nodes - (args.num_nodes % args.batch_size)
    for his_s, _, true_s, _, _ in generate_batch(graphs, [(i, 0, 0, 0) for i in range(tmp, args.num_nodes)],
                                                 args.target_snapshot-args.historical_len,
                                                 args.num_nodes-tmp, is_train=False, is_cuda=True):
        with torch.no_grad():
            pred, _ = net(his_s)
            probs.append(pred.cpu().numpy())
probs = np.concatenate(probs, axis=0)
print(np.max(probs))
# to avoid self-loop
graph = graphs[args.target_snapshot]
xindices, yindices = graph.nonzero()
pos_links = set([(x, y) for x, y in zip(xindices, yindices)])
neg_links = []
while len(neg_links) < len(pos_links):
    row = random.randint(0, args.num_nodes-1)
    col = random.randint(0, args.num_nodes-1)
    if (row, col) not in pos_links:
        neg_links.append((row, col))
pos_links = list(pos_links)
y_true = [1 for _ in range(len(pos_links))] + [0 for _ in range(len(neg_links))]
y_pred = [probs[x, y] for x, y in pos_links + neg_links]
auc = roc_auc_score(y_true, y_pred)
ap = average_precision_score(y_true, y_pred)

tp_links = [(x, y) for x, y in pos_links if probs[x, y] >= 0.5]
predicted_graph = csr_matrix(np.where(probs >= 0.5, 1, 0))
tmp_graph = predicted_graph - graph
num_err_links = np.sum(np.abs(tmp_graph.data))

print('AUC: {:.4f}'.format(auc))
print('AP: {:.4f}'.format(ap))
print('Err Links: {:.4f}'.format(num_err_links))
print('True links: {}/TP links: {}'.format(int(graph.sum()), len(tp_links)))

# find target links to perform attack
# top100 links with the highest existing probability
tp_link_probs = [probs[x, y] for x, y in tp_links]
top_100_pf_links = [tp_links[i] for i in np.argsort(tp_link_probs)[-100:]]
file_path = 'results/{}/tp/{}/'.format(args.dataset, args.historical_len)
if not os.path.exists(file_path):
    os.makedirs(file_path)
with open('{}/{}_target_links.list'.format(file_path, args.target_snapshot), 'wb') as f:
    pkl.dump(top_100_pf_links, f)

# top100 links with the highest edge betweenness
file_path = 'results/{}/ebc/{}/'.format(args.dataset, args.historical_len)
if not os.path.exists(file_path):
    os.makedirs(file_path)
print('Calculating edge between centrality:')
st = time.time()
ebc = nx.edge_betweenness_centrality(nx.DiGraph(graph))
print('Cost time: {:.2f} seconds.'.format(time.time() - st))
ebc = [ebc[(s, t)] for s, t in tp_links]
top_100_ebc_indices = list(np.argsort(ebc))[-100:]
top_100_ebc_links = [tp_links[idx] for idx in top_100_ebc_indices]
with open('{}/{}_target_links.list'.format(file_path, args.target_snapshot), 'wb') as f:
    pkl.dump(top_100_ebc_links, f)

# top100 links with the highest degree centrality (degree of source node + degree of target node)
# Calculating Degree Centrality
file_path = 'results/{}/dc/{}/'.format(args.dataset, args.historical_len)
if not os.path.exists(file_path):
    os.makedirs(file_path)
st = time.time()
print('Calculating out degree centrality:')
dc = [graph[s].sum() + graph[t].sum() for s, t in tp_links]
print('Cost time: {:.2f} seconds.'.format(time.time() - st))
top_100_dc_indices = list(np.argsort(dc))[-100:]
top_100_dc_links = [tp_links[idx] for idx in top_100_dc_indices]
with open('{}/{}_target_links.list'.format(file_path, args.target_snapshot), 'wb') as f:
    pkl.dump(top_100_dc_links, f)
