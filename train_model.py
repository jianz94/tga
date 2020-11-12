# coding: utf-8
import os
import torch.optim as optim
from utils import *
from config import args, device
from model import ddne, StructuralLoss, ConnectionLoss
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

graphs, node_pairs = load_data(args.dataset, args.historical_len)
train_node_pairs_idx = random.sample(range(len(node_pairs)), int(len(node_pairs)*0.8))
val_node_pairs_idx = list(set(range(len(node_pairs))) - set(train_node_pairs_idx))
train_node_pairs = [node_pairs[i] for i in train_node_pairs_idx]
val_node_pairs = [node_pairs[i] for i in val_node_pairs_idx]

with open('models/{}.model'.format(args.dataset), 'r') as f:
    model_definition = [int(x) for x in f.readline().split(',')]
args.num_nodes = model_definition[-1]
net = ddne(model_definition[0], model_definition[1:]).to(device)
net.apply(weights_init)
criterion_structural = StructuralLoss(args.alpha)
criterion_connection = ConnectionLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.001)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

for epoch in range(args.num_epochs):
    # training
    loss = 0
    net.train()
    pbar = tqdm(generate_batch(graphs, train_node_pairs,
                               args.target_snapshot - args.historical_len,
                               args.batch_size, is_cuda=True),
                total=len(train_node_pairs) // args.batch_size)
    pbar.set_description('Epoch: {}-Train'.format(epoch))
    for his_s, his_t, true_s, true_t, weight in pbar:
        optimizer.zero_grad()
        pred_s, cs = net(his_s)
        pred_t, ct = net(his_t)
        ls = criterion_structural(true_s, pred_s) + criterion_structural(true_t, pred_t)
        lc = criterion_connection(cs, ct, weight)
        loss = ls + args.beta*lc
        loss.backward()
        optimizer.step()
        
    # validation
    net.eval()
    pbar = tqdm(generate_batch(graphs, val_node_pairs,
                               args.target_snapshot - args.historical_len,
                               args.batch_size, False, is_cuda=True),
                total=len(val_node_pairs) // args.batch_size)
    pbar.set_description('Epoch: {}-Validation'.format(epoch))
    with torch.no_grad():
        val_loss = []
        for his_s, his_t, true_s, true_t, weight in pbar:
            pred_s, cs = net(his_s)
            pred_t, ct = net(his_t)
            ls = criterion_structural(true_s, pred_s) + criterion_structural(true_t, pred_t)
            lc = criterion_connection(cs, ct, weight)
            val_loss.append(ls.item() + args.beta*lc.item())
    print('Epoch {} Train Loss: {:.4f} Val Loss: {:.4f}'.format(epoch, loss.detach().item(), np.mean(val_loss)))

    scheduler.step()

torch.save(net.state_dict(), 'models/{}_len_{}_params.pkl'.format(args.dataset, args.historical_len))

