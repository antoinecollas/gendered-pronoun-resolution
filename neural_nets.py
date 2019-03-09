import torch, sys
import torch.nn as nn
from torch.nn.functional import softmax

class Pooling(nn.Module):
    def __init__(self, in_features, d_proj=256):
        super(Pooling, self).__init__()
        self.proj_pronoun = nn.Linear(in_features, d_proj)
        self.proj_other = nn.Linear(in_features, d_proj)
        self.att_pronoun = nn.Linear(d_proj, 1)
        self.att_other = nn.Linear(d_proj, 1)

    def forward(self, x):
        pronoun, A, B = x
        for i in range(len(pronoun)):
            pronoun[i] = self.proj_pronoun(pronoun[i])
            A[i] = self.proj_other(A[i])
            B[i] = self.proj_other(B[i])
        
        weights_pronoun, weights_A, weights_B = list(), list(), list()
        for i in range(len(pronoun)):
            weights_pronoun.append(softmax(self.att_pronoun(pronoun[i]).reshape(-1), dim=0))
            weights_A.append(softmax(self.att_other(A[i]).reshape(-1), dim=0))
            weights_B.append(softmax(self.att_other(B[i]).reshape(-1), dim=0))
        
        for i in range(len(pronoun)):
            pronoun[i] = torch.sum(pronoun[i]*weights_pronoun[i].unsqueeze(1), dim=0)
            A[i] = torch.sum(A[i]*weights_A[i].unsqueeze(1), dim=0)
            B[i] = torch.sum(B[i]*weights_B[i].unsqueeze(1), dim=0)

        pronoun = torch.stack(pronoun)
        A = torch.stack(A)
        B = torch.stack(B)
        
        return pronoun, A, B

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
            nn.Linear(d_hid, d_hid),
            nn.Tanh(),
            nn.BatchNorm1d(d_hid),
            nn.Dropout(dropout),
            nn.Linear(d_hid, out_features),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x