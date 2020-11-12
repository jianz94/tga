# coding: utf-8
import torch
import torch.nn as nn
from config import args


class ddne(nn.Module):
    def __init__(self, encoder, decoders):
        super(ddne, self).__init__()
        self.h_len = args.historical_len
        self.embedding = Embedding(args.num_nodes, encoder)
        self.encoder = nn.GRU(input_size=encoder, hidden_size=encoder, num_layers=1, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(2*args.historical_len*encoder, decoders[0]),
            nn.ReLU(),
        )
        for i in range(1, len(decoders)):
            self.decoder.add_module('decoder_{}'.format(i), nn.Linear(decoders[i-1], decoders[i]))
            if i == len(decoders) - 1:
                self.decoder.add_module('decoder_ac_{}'.format(i), nn.Sigmoid())
            else:
                self.decoder.add_module('decoder_ac_{}'.format(i), nn.ReLU())
        self.decoder = self.decoder
        assert decoders[-1] == args.num_nodes, 'The dimension of the last decoder should equals to the number of nodes.'

    def forward(self, x):
        def encoding(seq):
            activations = [self.encoder(self.embedding(seq[0]))[0]]
            for i, nr in enumerate(seq[1:]):
                activations.append(self.encoder(torch.add(activations[i - 1], self.embedding(nr)))[0])
            return torch.cat(activations, dim=2)
        hr = encoding(x)
        hr = hr.reshape((hr.size()[0], -1)).squeeze(dim=1)
        hl = encoding(x[[idx for idx in range(args.historical_len-1, -1, -1)]])
        hl = hl.reshape((hl.size()[0], -1)).squeeze(dim=1)
        c = torch.cat([hl, hr], dim=1)
        output = self.decoder(c)

        return output, c


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, embedding_dim))

    def forward(self, x):
        return torch.matmul(x.squeeze(1), self.weight).unsqueeze(1)


class StructuralLoss(nn.Module):
    def __init__(self, alpha):
        self.alpha = alpha
        super(StructuralLoss, self).__init__()

    def forward(self, y_true, y_pred):
        z = torch.ones_like(y_true)
        z = torch.add(z, torch.mul(y_true, self.alpha))
        return nn.BCELoss(weight=z)(y_pred, y_true)


class ConnectionLoss(nn.Module):
    def __init__(self):
        super(ConnectionLoss, self).__init__()

    def forward(self, ci, cj, nij):
        return torch.mean(torch.mul(nij, torch.sum(torch.pow(torch.sub(ci, cj), 2), dim=1)))


class MaskedLoss(nn.Module):
    def __init__(self):
        super(MaskedLoss, self).__init__()

    def forward(self, y_true, y_pred, target):
        mask = torch.zeros_like(y_true).to(y_true)
        mask_idx = (torch.LongTensor([0]), torch.LongTensor([target]))
        mask = mask.index_put_(mask_idx, torch.FloatTensor([1]).to(y_true))
        # return -nn.BCELoss(weight=mask)(y_pred, y_true)
        return -torch.sum(torch.pow(torch.mul(torch.sub(y_pred, y_true), mask), 2))


if __name__ == '__main__':
    net = ddne([128, 151])
    inputs = torch.randn(2, 1, 1, 151)
    output, c = net(inputs)
    print(output.shape, c.shape)








