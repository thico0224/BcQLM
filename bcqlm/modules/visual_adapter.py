import torch.nn as nn

class VisualAdapter(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mapper = nn.Linear(input_dim, output_dim)

    def forward(self, features):
        return self.mapper(features)