import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def clone_layer(module, N):
    return [copy.deepcopy(module) for _ in range(N)]


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        hid_dim = args.hid_dim
        self.model = nn.Sequential(
            nn.Conv1d(3, hid_dim // 2, 1),
            nn.GroupNorm(8, hid_dim // 2),
            nn.GELU(),
            nn.Conv1d(hid_dim // 2, hid_dim, 1),
            nn.GroupNorm(8, hid_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # B,D,S
        x = self.model(x)
        x = x.permute(0, 2, 1)  # B,S,D
        return x


class PCTransformer(nn.Module):
    def __init__(self, args):
        super(PCTransformer, self).__init__()
        hid_dim = args.hid_dim
        nhead = args.nhead
        dropout = args.dropout
        dim_fc = args.dim_fc
        n_attn = args.n_attn

        self.encoder = Encoder(args)
        model = clone_layer(nn.TransformerEncoderLayer(d_model=hid_dim,
                                                       nhead=nhead,
                                                       dim_feedforward=dim_fc,
                                                       dropout=dropout,
                                                       activation='gelu'), n_attn)
        self.model = nn.Sequential(*model)
        self.decoder = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.GroupNorm(8, hid_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim // 2, args.num_cls)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # S,B,D
        x = self.model(x)
        x = x.permute(1, 0, 2)  # B,S,D
        x = torch.mean(x, dim=1)
        return self.decoder(x)
