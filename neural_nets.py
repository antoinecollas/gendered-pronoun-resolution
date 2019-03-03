import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, d_hid = 512):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(in_features), 
            nn.Dropout(dropout),
            nn.Linear(in_features, d_hid),
            nn.Tanh(), 
            nn.BatchNorm1d(d_hid), 
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_hid), nn.Tanh(),
            nn.Tanh(),
            nn.BatchNorm1d(d_hid),
            nn.Dropout(dropout),
            nn.Linear(d_hid, out_features),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x