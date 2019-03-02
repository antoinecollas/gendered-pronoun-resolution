import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        hidden_size = [4096, 1024, 256]
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_size[0]),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1]),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_size[1], out_features=hidden_size[2]),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_size[2], out_features=out_features),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x