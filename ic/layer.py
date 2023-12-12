# import torch
# from torch import nn

# class Adapter(nn.Module):
#   """Layers used in mapping text embeddings to visual outputs."""

#   def __init__(self, in_dim: int, out_dim: int):
#     super().__init__()

#     self.model = nn.Linear(in_dim, out_dim)

#   def forward(self, x: torch.Tensor) -> torch.Tensor:
#     outputs = self.model(x)
#     return outputs  # (N, T, D)